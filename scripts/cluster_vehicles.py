"""
Vehicle Image Clustering & LLM Labeling Pipeline

Clusters ~52k existing vehicle images using DINOv2 embeddings + HDBSCAN,
then labels cluster representatives with OpenAI GPT-5.2 vision.

Usage:
    uv run --extra clustering --extra db --extra storage python scripts/cluster_vehicles.py run-all
    uv run --extra clustering --extra db --extra storage python scripts/cluster_vehicles.py download
    uv run --extra clustering --extra db --extra storage python scripts/cluster_vehicles.py embed
    uv run --extra clustering --extra db --extra storage python scripts/cluster_vehicles.py cluster
    uv run --extra clustering --extra db --extra storage python scripts/cluster_vehicles.py label
"""

import base64
import json
import logging
import os
import time
from pathlib import Path

import click
import numpy as np
from minio import Minio
from PIL import Image
from sqlalchemy import JSON
from sqlalchemy import BigInteger
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import LargeBinary
from sqlalchemy import String
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 384  # DINOv2-small CLS token dimension
EMBEDDING_BYTES = EMBEDDING_DIM * 4  # float32 = 4 bytes each

VALID_VEHICLE_TYPES = {
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
    "unknown",
}

# ---------------------------------------------------------------------------
# Lightweight ORM models — intentional mirrors of stopsign/database.py
# to avoid importing stopsign.telemetry (OpenTelemetry).
# Keep in sync: VehicleEmbedding, VehicleAttribute columns must match.
# ---------------------------------------------------------------------------
Base = declarative_base()


class VehiclePass(Base):
    __tablename__ = "vehicle_passes"
    id = Column(BigInteger, primary_key=True)
    timestamp = Column(DateTime)
    image_path = Column(String)


class VehicleEmbedding(Base):
    __tablename__ = "vehicle_embeddings"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    vehicle_pass_id = Column(BigInteger, ForeignKey("vehicle_passes.id"), unique=True, nullable=False)
    embedding = Column(LargeBinary, nullable=False)
    model_name = Column(String(64), nullable=False)
    created_at = Column(DateTime, default=func.now())


class VehicleAttribute(Base):
    __tablename__ = "vehicle_attributes"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    vehicle_pass_id = Column(BigInteger, ForeignKey("vehicle_passes.id"), unique=True, nullable=False)
    cluster_id = Column(Integer)
    vehicle_type = Column(String(64))
    color = Column(String(64))
    make_model = Column(String(128))
    confidence = Column(Float)
    is_representative = Column(Boolean, default=False)
    raw_llm_response = Column(JSON)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_db_url() -> str:
    url = os.environ.get("DB_URL")
    if not url:
        raise click.ClickException("DB_URL environment variable is required")
    return url


def make_session(db_url: str):
    engine = create_engine(db_url, pool_pre_ping=True)
    # Create new tables only (won't touch existing ones)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine), engine


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
        # Legacy paths: minio://vehicle-images/filename.jpg
        # Images were migrated to Bremen, extract just the filename
        parts = image_path.split("/", 3)
        if len(parts) >= 4:
            return "bremen", bucket, parts[3]
        raise ValueError(f"Malformed minio:// path: {image_path}")
    else:
        raise ValueError(f"Unknown image_path scheme: {image_path}")


def embedding_to_bytes(arr: np.ndarray) -> bytes:
    """Pack float32 numpy array into little-endian bytes."""
    return arr.astype("<f4").tobytes()


def bytes_to_embedding(data: bytes, expected_dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Unpack bytes into float32 numpy array with shape validation."""
    expected_bytes = expected_dim * 4
    if len(data) != expected_bytes:
        raise ValueError(f"Embedding byte length {len(data)} != expected {expected_bytes} ({expected_dim} x float32)")
    return np.frombuffer(data, dtype="<f4")


def validate_label_result(result: dict) -> dict:
    """Validate and sanitize an LLM label response."""
    vehicle_type = result.get("vehicle_type", "unknown")
    if vehicle_type not in VALID_VEHICLE_TYPES:
        vehicle_type = "unknown"

    confidence = result.get("confidence", 0.0)
    try:
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "vehicle_type": vehicle_type,
        "color": str(result.get("color", "unknown"))[:64],
        "make_model": str(result.get("make_model", "unknown"))[:128],
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Stage 1: Download
# ---------------------------------------------------------------------------


@click.group()
def cli():
    """Vehicle image clustering & LLM labeling pipeline."""
    pass


@cli.command()
@click.option("--cache-dir", default="./image_cache", help="Local directory for cached images")
@click.option("--limit", default=0, help="Max images to download (0 = all)")
def download(cache_dir: str, limit: int):
    """Stage 1: Download vehicle images from MinIO to local cache."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    db_url = get_db_url()
    Session, _engine = make_session(db_url)

    with Session() as session:
        query = session.query(VehiclePass.id, VehiclePass.image_path).filter(
            VehiclePass.image_path.isnot(None),
            VehiclePass.image_path != "",
        )
        passes = query.all()

    logger.info(f"Found {len(passes)} passes with images")

    # Filter to supported schemes
    valid = []
    for p in passes:
        if p.image_path and any(p.image_path.startswith(s) for s in ("bremen://", "local://", "minio://")):
            valid.append(p)
    logger.info(f"Valid image paths: {len(valid)}")

    if limit > 0:
        valid = valid[:limit]

    # All images resolve to Bremen (including legacy minio:// paths)
    bremen_client = get_bremen_client()
    downloaded = 0
    skipped = 0
    errors = 0

    for p in tqdm(valid, desc="Downloading"):
        filename = f"{p.id}.jpg"
        dest = cache / filename
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            continue
        # Remove 0-byte partial files
        if dest.exists():
            dest.unlink()

        try:
            _scheme, bucket, obj_name = resolve_image_object(p.image_path)
            bremen_client.fget_object(bucket, obj_name, str(dest))
            downloaded += 1
        except Exception as e:
            errors += 1
            if errors <= 10:
                logger.warning(f"Failed to download pass {p.id}: {e}")
            elif errors == 11:
                logger.warning("Suppressing further download errors...")

    logger.info(f"Download complete: {downloaded} new, {skipped} cached, {errors} errors")


# ---------------------------------------------------------------------------
# Stage 2: Embed
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--cache-dir", default="./image_cache", help="Local image cache directory")
@click.option("--batch-size", default=64, help="Batch size for embedding")
@click.option("--model-name", default="dinov2-small", help="Model identifier stored in DB")
def embed(cache_dir: str, batch_size: int, model_name: str):
    """Stage 2: Generate DINOv2 embeddings for cached images."""
    import torch
    from transformers import AutoImageProcessor
    from transformers import AutoModel

    cache = Path(cache_dir)
    if not cache.exists():
        raise click.ClickException(f"Cache dir {cache_dir} does not exist. Run 'download' first.")

    db_url = get_db_url()
    Session, _engine = make_session(db_url)

    # Find passes that need embedding — scoped by model_name
    with Session() as session:
        already_embedded = {
            r[0]
            for r in session.query(VehicleEmbedding.vehicle_pass_id)
            .filter(VehicleEmbedding.model_name == model_name)
            .all()
        }

    # List cached images and filter out already-embedded
    image_files = sorted(cache.glob("*.jpg"))
    to_embed = []
    for f in image_files:
        pass_id = int(f.stem)
        if pass_id not in already_embedded:
            to_embed.append((pass_id, f))

    logger.info(
        f"Total cached: {len(image_files)}, already embedded ({model_name}): {len(already_embedded)}, "
        f"to embed: {len(to_embed)}"
    )

    if not to_embed:
        logger.info("Nothing to embed.")
        return

    # Load model
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Loading DINOv2-small on {device}...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
    model.eval()

    # Process in batches
    for batch_start in tqdm(range(0, len(to_embed), batch_size), desc="Embedding batches"):
        batch = to_embed[batch_start : batch_start + batch_size]
        images = []
        valid_items = []

        for pass_id, fpath in batch:
            try:
                with Image.open(fpath) as img:
                    img_rgb = img.convert("RGB")
                images.append(img_rgb)
                valid_items.append((pass_id, fpath))
            except Exception as e:
                logger.warning(f"Failed to open image {fpath}: {e}")

        if not images:
            continue

        # Run inference
        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt").to(device)
            outputs = model(**inputs)
            # Use CLS token embedding (first token)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # Store in DB
        with Session() as session:
            for i, (pass_id, _) in enumerate(valid_items):
                emb = VehicleEmbedding(
                    vehicle_pass_id=pass_id,
                    embedding=embedding_to_bytes(embeddings[i]),
                    model_name=model_name,
                )
                session.merge(emb)
            session.commit()

    with Session() as session:
        total = session.query(VehicleEmbedding).filter(VehicleEmbedding.model_name == model_name).count()
    logger.info(f"Embedding complete. Total embeddings in DB ({model_name}): {total}")


# ---------------------------------------------------------------------------
# Stage 3: Cluster
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--min-cluster-size", default=5, help="HDBSCAN min_cluster_size")
@click.option("--min-samples", default=3, help="HDBSCAN min_samples")
@click.option("--umap-dims", default=15, help="UMAP target dimensions (0 to skip)")
@click.option("--umap-neighbors", default=30, help="UMAP n_neighbors")
@click.option("--model-name", default="dinov2-small", help="Filter embeddings by model name")
def cluster(min_cluster_size: int, min_samples: int, umap_dims: int, umap_neighbors: int, model_name: str):
    """Stage 3: Cluster embeddings with UMAP + HDBSCAN."""
    import hdbscan

    db_url = get_db_url()
    Session, engine = make_session(db_url)

    # Load embeddings filtered by model_name
    with Session() as session:
        rows = (
            session.query(
                VehicleEmbedding.vehicle_pass_id,
                VehicleEmbedding.embedding,
            )
            .filter(VehicleEmbedding.model_name == model_name)
            .all()
        )

    if not rows:
        raise click.ClickException(f"No embeddings found for model '{model_name}'. Run 'embed' first.")

    logger.info(f"Loaded {len(rows)} embeddings (model={model_name})")

    pass_ids = np.array([r.vehicle_pass_id for r in rows])
    embeddings = np.array([bytes_to_embedding(r.embedding) for r in rows])

    logger.info(f"Embedding matrix shape: {embeddings.shape}")

    # UMAP dimensionality reduction (critical for HDBSCAN on high-dim data)
    if umap_dims > 0:
        import umap

        logger.info(f"Running UMAP ({embeddings.shape[1]}d -> {umap_dims}d, n_neighbors={umap_neighbors})...")
        reducer = umap.UMAP(
            n_components=umap_dims,
            n_neighbors=umap_neighbors,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
        cluster_data = reducer.fit_transform(embeddings)
        logger.info(f"UMAP output shape: {cluster_data.shape}")
    else:
        cluster_data = embeddings

    # Run HDBSCAN
    logger.info(f"Running HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(cluster_data)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    noise_pct = n_noise / len(labels) * 100

    logger.info(f"Clusters: {n_clusters}, Noise: {n_noise} ({noise_pct:.1f}%)")

    # Cluster size distribution
    from collections import Counter

    sizes = Counter(label for label in labels if label != -1)
    if sizes:
        sorted_sizes = sorted(sizes.values(), reverse=True)
        median = sorted_sizes[len(sorted_sizes) // 2]
        logger.info(f"Cluster sizes — min: {sorted_sizes[-1]}, max: {sorted_sizes[0]}, median: {median}")

    # Find representative for each cluster (closest to centroid)
    representatives = {}  # cluster_id -> pass_id
    for cid in set(labels):
        if cid == -1:
            continue
        mask = labels == cid
        cluster_embeddings = embeddings[mask]
        cluster_pass_ids = pass_ids[mask]
        centroid = cluster_embeddings.mean(axis=0)
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        best_idx = np.argmin(distances)
        representatives[cid] = int(cluster_pass_ids[best_idx])

    logger.info(f"Selected {len(representatives)} cluster representatives")
    rep_set = set(representatives.values())

    # Upsert to DB — preserves existing labels (vehicle_type, color, etc.)
    BATCH = 2000
    items = list(zip(pass_ids, labels))
    for batch_start in tqdm(range(0, len(items), BATCH), desc="Writing attributes"):
        batch = items[batch_start : batch_start + BATCH]
        with Session() as session:
            for pid, cid in batch:
                pid = int(pid)
                cid = int(cid)
                is_rep = pid in rep_set
                stmt = (
                    pg_insert(VehicleAttribute)
                    .values(
                        vehicle_pass_id=pid,
                        cluster_id=cid if cid != -1 else None,
                        is_representative=is_rep,
                    )
                    .on_conflict_do_update(
                        index_elements=["vehicle_pass_id"],
                        set_={
                            "cluster_id": cid if cid != -1 else None,
                            "is_representative": is_rep,
                            "updated_at": func.now(),
                        },
                    )
                )
                session.execute(stmt)
            session.commit()

    logger.info("Cluster results saved to vehicle_attributes (labels preserved)")


# ---------------------------------------------------------------------------
# Stage 4: Label
# ---------------------------------------------------------------------------

VISION_PROMPT = """Analyze this cropped vehicle image. Return a JSON object with:
{
  "vehicle_type": "sedan|suv|pickup|van|hatchback|coupe|wagon|motorcycle|bus|truck|unknown",
  "color": "primary color of the vehicle",
  "make_model": "best guess at make and model, or 'unknown'",
  "confidence": 0.0-1.0
}
Only return the JSON, no other text."""

MAX_LABEL_RETRIES = 2


@cli.command()
@click.option("--cache-dir", default="./image_cache", help="Local image cache directory")
@click.option("--max-clusters", default=0, help="Max clusters to label (0 = all)")
@click.option("--model", default="gpt-5.2", help="OpenAI vision model")
def label(cache_dir: str, max_clusters: int, model: str):
    """Stage 4: Label cluster representatives with OpenAI vision."""
    from openai import OpenAI

    cache = Path(cache_dir)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise click.ClickException("OPENAI_API_KEY environment variable is required")

    client = OpenAI(api_key=api_key)

    db_url = get_db_url()
    Session, _engine = make_session(db_url)

    # Find unlabeled representatives, ordered deterministically
    with Session() as session:
        reps = (
            session.query(VehicleAttribute)
            .filter(
                VehicleAttribute.is_representative.is_(True),
                VehicleAttribute.vehicle_type.is_(None),
            )
            .order_by(VehicleAttribute.cluster_id)
            .all()
        )
        # Detach from session
        rep_data = [(r.id, r.vehicle_pass_id, r.cluster_id) for r in reps]

    logger.info(f"Found {len(rep_data)} unlabeled representatives")

    if max_clusters > 0:
        rep_data = rep_data[:max_clusters]

    labeled = 0
    errors = 0

    for attr_id, pass_id, cluster_id in tqdm(rep_data, desc="Labeling"):
        image_path = cache / f"{pass_id}.jpg"
        if not image_path.exists():
            logger.warning(f"Image not found for pass {pass_id}, skipping")
            errors += 1
            continue

        # Retry loop for API/parsing failures
        result = None
        for attempt in range(1, MAX_LABEL_RETRIES + 2):
            try:
                # Encode image as base64
                with open(image_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")

                response = client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": VISION_PROMPT},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                                },
                            ],
                        }
                    ],
                )

                raw_text = response.choices[0].message.content.strip()
                parsed = json.loads(raw_text)
                result = validate_label_result(parsed)
                break  # Success

            except (json.JSONDecodeError, KeyError) as e:
                if attempt <= MAX_LABEL_RETRIES:
                    logger.warning(f"Parse error for pass {pass_id} (attempt {attempt}): {e}, retrying...")
                    time.sleep(1.0 * attempt)
                else:
                    logger.warning(f"Failed to parse LLM response for pass {pass_id} after {attempt} attempts: {e}")
            except Exception as e:
                if attempt <= MAX_LABEL_RETRIES:
                    logger.warning(f"API error for pass {pass_id} (attempt {attempt}): {e}, retrying...")
                    time.sleep(2.0 * attempt)
                else:
                    logger.warning(f"Failed to label pass {pass_id} after {attempt} attempts: {e}")

        if result is None:
            errors += 1
            continue

        # Update representative
        with Session() as session:
            session.query(VehicleAttribute).filter(VehicleAttribute.id == attr_id).update(
                {
                    "vehicle_type": result["vehicle_type"],
                    "color": result["color"],
                    "make_model": result["make_model"],
                    "confidence": result["confidence"],
                    "raw_llm_response": parsed,
                    "updated_at": func.now(),
                }
            )

            # Propagate to all cluster members
            if cluster_id is not None:
                session.query(VehicleAttribute).filter(
                    VehicleAttribute.cluster_id == cluster_id,
                    VehicleAttribute.is_representative.is_(False),
                ).update(
                    {
                        "vehicle_type": result["vehicle_type"],
                        "color": result["color"],
                        "make_model": result["make_model"],
                        "confidence": result["confidence"],
                        "updated_at": func.now(),
                    }
                )
            session.commit()

        labeled += 1

    logger.info(f"Labeling complete: {labeled} labeled, {errors} errors")

    # Summary
    with Session() as session:
        rows = session.execute(
            text(
                "SELECT vehicle_type, COUNT(*) as cnt FROM vehicle_attributes "
                "WHERE vehicle_type IS NOT NULL GROUP BY vehicle_type ORDER BY cnt DESC"
            )
        )
        logger.info("Vehicle type distribution:")
        for row in rows:
            logger.info(f"  {row[0]}: {row[1]}")


# ---------------------------------------------------------------------------
# Run All
# ---------------------------------------------------------------------------


@cli.command("run-all")
@click.option("--cache-dir", default="./image_cache", help="Local image cache directory")
@click.option("--batch-size", default=64, help="Embedding batch size")
@click.option("--min-cluster-size", default=5, help="HDBSCAN min_cluster_size")
@click.option("--min-samples", default=3, help="HDBSCAN min_samples")
@click.option("--umap-dims", default=15, help="UMAP target dimensions (0 to skip)")
@click.option("--umap-neighbors", default=30, help="UMAP n_neighbors")
@click.option("--max-clusters", default=0, help="Max clusters to label (0 = all)")
@click.option("--model", default="gpt-5.2", help="OpenAI vision model")
@click.option("--model-name", default="dinov2-small", help="Embedding model identifier")
@click.pass_context
def run_all(
    ctx,
    cache_dir,
    batch_size,
    min_cluster_size,
    min_samples,
    umap_dims,
    umap_neighbors,
    max_clusters,
    model,
    model_name,
):
    """Run all 4 stages sequentially."""
    ctx.invoke(download, cache_dir=cache_dir)
    ctx.invoke(embed, cache_dir=cache_dir, batch_size=batch_size, model_name=model_name)
    ctx.invoke(
        cluster,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        umap_dims=umap_dims,
        umap_neighbors=umap_neighbors,
        model_name=model_name,
    )
    ctx.invoke(label, cache_dir=cache_dir, max_clusters=max_clusters, model=model)


if __name__ == "__main__":
    cli()
