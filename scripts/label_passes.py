"""
Label-Everything Pipeline

Labels every vehicle_passes image with an OpenAI vision model (structured
output, strict json_schema) and writes the results to a new `vehicle_labels`
table (does NOT touch vehicle_attributes / the cluster pipeline).

Standalone script style mirrors scripts/cluster_vehicles.py: no import of
stopsign.settings or stopsign.telemetry, raw SQL for the new table, its own
copies of the Bremen MinIO / image-path-resolution helpers.

Usage:
    uv run --extra db --extra storage python scripts/label_passes.py
    uv run --extra db --extra storage python scripts/label_passes.py --limit 200 --dry-run
    uv run --extra db --extra storage python scripts/label_passes.py --pass-ids 123,456,789
"""

import asyncio
import base64
import json
import logging
import os
import time

import click
from minio import Minio
from openai import APIConnectionError
from openai import APIStatusError
from openai import APITimeoutError
from openai import AsyncOpenAI
from openai import RateLimitError
from sqlalchemy import Engine
from sqlalchemy import create_engine
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Closed vocabularies (enforced via OpenAI structured output json_schema)
# ---------------------------------------------------------------------------
VEHICLE_TYPES = [
    "sedan",
    "suv",
    "pickup",
    "van",
    "hatchback",
    "coupe",
    "wagon",
    "motorcycle",
    "bus",
    "truck",
    "bicycle",
    "other",
    "unknown",
]

COLORS = [
    "white",
    "black",
    "silver",
    "gray",
    "red",
    "blue",
    "green",
    "brown",
    "beige",
    "gold",
    "yellow",
    "orange",
    "purple",
    "unknown",
]

MAKES = [
    "toyota",
    "honda",
    "ford",
    "chevrolet",
    "nissan",
    "hyundai",
    "kia",
    "subaru",
    "volkswagen",
    "bmw",
    "mercedes_benz",
    "audi",
    "lexus",
    "mazda",
    "jeep",
    "ram",
    "gmc",
    "dodge",
    "tesla",
    "volvo",
    "acura",
    "infiniti",
    "buick",
    "cadillac",
    "other",
    "unknown",
]

DEFAULT_MODEL = "gpt-5.6-luna"
PROMPT_VERSION = 1

# $/token, used only for the running cost estimate in progress logs.
PRICE_PER_INPUT_TOKEN = 0.20 / 1_000_000
PRICE_PER_OUTPUT_TOKEN = 1.20 / 1_000_000

SYSTEM_PROMPT = (
    "You label vehicle crops from a fixed traffic camera at a residential stop "
    "sign. Images are small and may be motion-blurred. Use 'unknown' when not "
    "discernible; use null for booleans/counts you cannot determine. model is "
    "the specific model name (e.g. 'CR-V', 'F-150') only when confident, else "
    "null. Confidences are honest 0-1 estimates. description: one short "
    "sentence, notable details only."
)

LABEL_JSON_SCHEMA = {
    "name": "vehicle_label",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "vehicle_type": {"type": "string", "enum": VEHICLE_TYPES},
            "color": {"type": "string", "enum": COLORS},
            "make": {"type": "string", "enum": MAKES},
            "model": {"type": ["string", "null"]},
            "roof_rack": {"type": ["boolean", "null"]},
            "commercial": {"type": ["boolean", "null"]},
            "damage_visible": {"type": ["boolean", "null"]},
            "headlights_on": {"type": ["boolean", "null"]},
            "passengers_visible": {"type": ["integer", "null"]},
            "type_confidence": {"type": "number"},
            "color_confidence": {"type": "number"},
            "make_model_confidence": {"type": "number"},
            "description": {"type": "string"},
        },
        "required": [
            "vehicle_type",
            "color",
            "make",
            "model",
            "roof_rack",
            "commercial",
            "damage_visible",
            "headlights_on",
            "passengers_visible",
            "type_confidence",
            "color_confidence",
            "make_model_confidence",
            "description",
        ],
        "additionalProperties": False,
    },
}

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS vehicle_labels (
    id BIGSERIAL PRIMARY KEY,
    vehicle_pass_id BIGINT NOT NULL UNIQUE REFERENCES vehicle_passes(id),
    vehicle_type TEXT NOT NULL,
    color TEXT NOT NULL,
    make TEXT NOT NULL,
    model TEXT,
    roof_rack BOOLEAN,
    commercial BOOLEAN,
    damage_visible BOOLEAN,
    headlights_on BOOLEAN,
    passengers_visible INTEGER,
    type_confidence REAL,
    color_confidence REAL,
    make_model_confidence REAL,
    description TEXT,
    llm_model TEXT NOT NULL,
    prompt_version INTEGER NOT NULL DEFAULT 1,
    tokens_in INTEGER,
    tokens_out INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_vehicle_labels_type ON vehicle_labels(vehicle_type);
CREATE INDEX IF NOT EXISTS idx_vehicle_labels_make ON vehicle_labels(make);
"""

UPSERT_SQL = text("""
INSERT INTO vehicle_labels (
    vehicle_pass_id, vehicle_type, color, make, model, roof_rack, commercial,
    damage_visible, headlights_on, passengers_visible, type_confidence,
    color_confidence, make_model_confidence, description, llm_model,
    prompt_version, tokens_in, tokens_out, updated_at
) VALUES (
    :vehicle_pass_id, :vehicle_type, :color, :make, :model, :roof_rack, :commercial,
    :damage_visible, :headlights_on, :passengers_visible, :type_confidence,
    :color_confidence, :make_model_confidence, :description, :llm_model,
    :prompt_version, :tokens_in, :tokens_out, now()
)
ON CONFLICT (vehicle_pass_id) DO UPDATE SET
    vehicle_type = EXCLUDED.vehicle_type,
    color = EXCLUDED.color,
    make = EXCLUDED.make,
    model = EXCLUDED.model,
    roof_rack = EXCLUDED.roof_rack,
    commercial = EXCLUDED.commercial,
    damage_visible = EXCLUDED.damage_visible,
    headlights_on = EXCLUDED.headlights_on,
    passengers_visible = EXCLUDED.passengers_visible,
    type_confidence = EXCLUDED.type_confidence,
    color_confidence = EXCLUDED.color_confidence,
    make_model_confidence = EXCLUDED.make_model_confidence,
    description = EXCLUDED.description,
    llm_model = EXCLUDED.llm_model,
    prompt_version = EXCLUDED.prompt_version,
    tokens_in = EXCLUDED.tokens_in,
    tokens_out = EXCLUDED.tokens_out,
    updated_at = now()
""")

MAX_API_ATTEMPTS = 5

# ---------------------------------------------------------------------------
# Env / connection helpers — copied/adapted from scripts/cluster_vehicles.py
# ---------------------------------------------------------------------------


def get_db_url() -> str:
    url = os.environ.get("DB_URL")
    if not url:
        raise click.ClickException("DB_URL environment variable is required")
    return url


def get_openai_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise click.ClickException("OPENAI_API_KEY environment variable is required")
    return key


def get_bremen_client() -> Minio:
    secret_key = os.environ.get("BREMEN_MINIO_SECRET_KEY")
    if not secret_key:
        raise click.ClickException("BREMEN_MINIO_SECRET_KEY environment variable is required")
    return Minio(
        os.environ.get("BREMEN_MINIO_ENDPOINT", "100.98.103.56:9000"),
        access_key=os.environ.get("BREMEN_MINIO_ACCESS_KEY", "root"),
        secret_key=secret_key,
        secure=False,
    )


def resolve_image_object(image_path: str) -> tuple[str, str, str]:
    """Return (scheme, bucket, object_name) from an image_path URI.

    All schemes resolve to Bremen MinIO — legacy minio:// images were
    migrated there but the DB paths were never updated.
    """
    bucket = os.environ.get("BREMEN_MINIO_BUCKET", "vehicle-images")
    if image_path.startswith("bremen://"):
        return "bremen", bucket, image_path.replace("bremen://", "")
    elif image_path.startswith("local://"):
        return "bremen", bucket, image_path.replace("local://", "")
    elif image_path.startswith("minio://"):
        parts = image_path.split("/", 3)
        if len(parts) >= 4:
            return "bremen", bucket, parts[3]
        raise ValueError(f"Malformed minio:// path: {image_path}")
    else:
        raise ValueError(f"Unknown image_path scheme: {image_path}")


def make_engine(db_url: str) -> Engine:
    engine = create_engine(db_url, pool_pre_ping=True, pool_size=20, max_overflow=10)
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
    return engine


# ---------------------------------------------------------------------------
# Sync helpers run inside asyncio.to_thread
# ---------------------------------------------------------------------------


def fetch_image_bytes_sync(minio_client: Minio, image_path: str) -> bytes:
    _scheme, bucket, obj_name = resolve_image_object(image_path)
    response = minio_client.get_object(bucket, obj_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def upsert_label_sync(engine: Engine, params: dict) -> None:
    with engine.begin() as conn:
        conn.execute(UPSERT_SQL, params)


def select_passes_sync(engine: Engine, limit: int, pass_ids: list[int] | None) -> list[tuple[int, str]]:
    with engine.connect() as conn:
        if pass_ids:
            rows = conn.execute(
                text("SELECT id, image_path FROM vehicle_passes WHERE id = ANY(:ids) ORDER BY id DESC"),
                {"ids": pass_ids},
            ).fetchall()
        else:
            sql = (
                "SELECT id, image_path FROM vehicle_passes "
                "WHERE image_path IS NOT NULL AND image_path != '' "
                "AND id NOT IN (SELECT vehicle_pass_id FROM vehicle_labels) "
                "ORDER BY id DESC"
            )
            if limit > 0:
                sql += " LIMIT :limit"
                rows = conn.execute(text(sql), {"limit": limit}).fetchall()
            else:
                rows = conn.execute(text(sql)).fetchall()
    return [(int(r[0]), r[1]) for r in rows]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class Stats:
    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.labeled = 0
        self.skipped_no_image = 0
        self.failed: list[tuple[int, str]] = []
        self.tokens_in = 0
        self.tokens_out = 0
        self.start_time = time.monotonic()
        self.lock = asyncio.Lock()

    @property
    def cost(self) -> float:
        return self.tokens_in * PRICE_PER_INPUT_TOKEN + self.tokens_out * PRICE_PER_OUTPUT_TOKEN

    async def record(self, *, labeled=False, skipped_no_image=False, failed: tuple[int, str] | None = None):
        async with self.lock:
            self.done += 1
            if labeled:
                self.labeled += 1
            if skipped_no_image:
                self.skipped_no_image += 1
            if failed is not None:
                self.failed.append(failed)
            if self.done % 50 == 0 or self.done == self.total:
                self._log_progress()

    def _log_progress(self):
        elapsed = time.monotonic() - self.start_time
        rate = self.done / elapsed if elapsed > 0 else 0.0
        remaining = self.total - self.done
        eta_s = remaining / rate if rate > 0 else float("inf")
        eta_str = f"{eta_s / 60:.1f}m" if eta_s != float("inf") else "unknown"
        logger.info(
            f"progress {self.done}/{self.total} ({rate:.2f}/s) "
            f"tokens_in={self.tokens_in} tokens_out={self.tokens_out} cost=${self.cost:.4f} "
            f"eta={eta_str} failures={len(self.failed)}"
        )


# ---------------------------------------------------------------------------
# OpenAI call with retry/backoff
# ---------------------------------------------------------------------------


def _retry_after_seconds(exc: Exception, attempt: int) -> float:
    response = getattr(exc, "response", None)
    if response is not None:
        header = response.headers.get("retry-after") if hasattr(response, "headers") else None
        if header:
            try:
                return float(header)
            except ValueError:
                pass
    return float(2**attempt)


async def label_image(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    model: str,
    detail: str,
    image_b64: str,
    pass_id: int,
) -> tuple[dict, int, int]:
    """Call OpenAI vision with the strict json_schema. Returns (parsed, tokens_in, tokens_out).

    One retry on schema-validation/parse failure. 429/5xx get exponential
    backoff honoring Retry-After, up to MAX_API_ATTEMPTS total attempts.
    """
    schema_retry_used = False
    attempt = 0
    last_exc: Exception | None = None

    while attempt < MAX_API_ATTEMPTS:
        attempt += 1
        try:
            async with sem:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}",
                                        "detail": detail,
                                    },
                                }
                            ],
                        },
                    ],
                    response_format={"type": "json_schema", "json_schema": LABEL_JSON_SCHEMA},
                )
            content = response.choices[0].message.content
            parsed = json.loads(content)
            usage = response.usage
            tokens_in = usage.prompt_tokens if usage else 0
            tokens_out = usage.completion_tokens if usage else 0
            return parsed, tokens_in, tokens_out
        except (RateLimitError, APIStatusError, APIConnectionError, APITimeoutError) as e:
            last_exc = e
            status_code = getattr(e, "status_code", None)
            transient = isinstance(e, RateLimitError) or isinstance(e, (APIConnectionError, APITimeoutError))
            transient = transient or (status_code is not None and status_code >= 500)
            if not transient or attempt >= MAX_API_ATTEMPTS:
                raise
            wait = _retry_after_seconds(e, attempt)
            logger.warning(
                f"pass {pass_id}: API error (attempt {attempt}/{MAX_API_ATTEMPTS}), retrying in {wait}s: {e}"
            )
            await asyncio.sleep(wait)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            last_exc = e
            if schema_retry_used:
                raise
            schema_retry_used = True
            logger.warning(f"pass {pass_id}: schema/parse error, retrying once: {e}")

    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


async def process_pass(
    pass_id: int,
    image_path: str,
    *,
    minio_client: Minio,
    openai_client: AsyncOpenAI | None,
    io_sem: asyncio.Semaphore,
    llm_sem: asyncio.Semaphore,
    engine: Engine,
    model: str,
    detail: str,
    dry_run: bool,
    stats: Stats,
):
    async with io_sem:
        try:
            image_bytes = await asyncio.to_thread(fetch_image_bytes_sync, minio_client, image_path)
        except Exception as e:
            logger.warning(f"pass {pass_id}: image fetch failed ({image_path}): {e}")
            await stats.record(skipped_no_image=True)
            return

    if dry_run:
        logger.info(f"[dry-run] pass {pass_id}: would label {len(image_bytes)} byte image ({image_path})")
        await stats.record(labeled=True)
        return

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    try:
        parsed, tokens_in, tokens_out = await label_image(openai_client, llm_sem, model, detail, image_b64, pass_id)
    except Exception as e:
        logger.warning(f"pass {pass_id}: labeling failed after retries: {e}")
        await stats.record(failed=(pass_id, str(e)))
        return

    async with stats.lock:
        stats.tokens_in += tokens_in
        stats.tokens_out += tokens_out

    params = {
        "vehicle_pass_id": pass_id,
        "vehicle_type": parsed.get("vehicle_type", "unknown"),
        "color": parsed.get("color", "unknown"),
        "make": parsed.get("make", "unknown"),
        "model": parsed.get("model"),
        "roof_rack": parsed.get("roof_rack"),
        "commercial": parsed.get("commercial"),
        "damage_visible": parsed.get("damage_visible"),
        "headlights_on": parsed.get("headlights_on"),
        "passengers_visible": parsed.get("passengers_visible"),
        "type_confidence": parsed.get("type_confidence"),
        "color_confidence": parsed.get("color_confidence"),
        "make_model_confidence": parsed.get("make_model_confidence"),
        "description": parsed.get("description"),
        "llm_model": model,
        "prompt_version": PROMPT_VERSION,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }

    async with io_sem:
        try:
            await asyncio.to_thread(upsert_label_sync, engine, params)
        except Exception as e:
            logger.warning(f"pass {pass_id}: DB upsert failed: {e}")
            await stats.record(failed=(pass_id, f"upsert failed: {e}"))
            return

    await stats.record(labeled=True)


async def run(limit: int, concurrency: int, model: str, pass_ids: list[int] | None, dry_run: bool, detail: str):
    db_url = get_db_url()
    openai_api_key = None if dry_run else get_openai_api_key()
    engine = make_engine(db_url)

    rows = await asyncio.to_thread(select_passes_sync, engine, limit, pass_ids)
    logger.info(f"Selected {len(rows)} passes to label (limit={limit or 'none'}, pass_ids={pass_ids or 'none'})")

    if not rows:
        logger.info("Nothing to do.")
        return

    minio_client = get_bremen_client()
    openai_client = AsyncOpenAI(api_key=openai_api_key) if not dry_run else None

    io_sem = asyncio.Semaphore(20)
    llm_sem = asyncio.Semaphore(concurrency)
    stats = Stats(total=len(rows))

    tasks = [
        process_pass(
            pass_id,
            image_path,
            minio_client=minio_client,
            openai_client=openai_client,
            io_sem=io_sem,
            llm_sem=llm_sem,
            engine=engine,
            model=model,
            detail=detail,
            dry_run=dry_run,
            stats=stats,
        )
        for pass_id, image_path in rows
    ]
    await asyncio.gather(*tasks)

    elapsed = time.monotonic() - stats.start_time
    logger.info("=" * 60)
    logger.info("Label run complete")
    logger.info(f"  labeled:           {stats.labeled}")
    logger.info(f"  skipped (no image): {stats.skipped_no_image}")
    logger.info(f"  failed:            {len(stats.failed)}")
    if stats.failed:
        failed_ids = [pid for pid, _ in stats.failed]
        logger.info(f"    failed pass_ids: {failed_ids}")
        for pid, err in stats.failed[:20]:
            logger.info(f"    pass {pid}: {err}")
    logger.info(f"  elapsed:           {elapsed:.1f}s")
    logger.info(f"  tokens_in:         {stats.tokens_in}")
    logger.info(f"  tokens_out:        {stats.tokens_out}")
    logger.info(f"  total cost:        ${stats.cost:.4f}")
    logger.info("=" * 60)


@click.command()
@click.option("--limit", default=0, type=int, help="Max passes to label (0 = all).")
@click.option("--concurrency", default=50, type=int, help="Max concurrent OpenAI calls.")
@click.option("--model", default=DEFAULT_MODEL, help="OpenAI vision model.")
@click.option("--pass-ids", default=None, help="Comma-separated vehicle_pass ids; overrides selection.")
@click.option("--dry-run", is_flag=True, default=False, help="Select + fetch images + build requests, no API calls.")
@click.option("--detail", type=click.Choice(["low", "auto"]), default="auto", help="Vision image detail level.")
def main(limit: int, concurrency: int, model: str, pass_ids: str | None, dry_run: bool, detail: str):
    """Label every unlabeled vehicle_passes image with an OpenAI vision model."""
    parsed_ids = None
    if pass_ids:
        parsed_ids = [int(x.strip()) for x in pass_ids.split(",") if x.strip()]

    asyncio.run(run(limit, concurrency, model, parsed_ids, dry_run, detail))


if __name__ == "__main__":
    main()
