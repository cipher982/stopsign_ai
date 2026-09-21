#!/usr/bin/env python
"""Watch a recorded clip offline.

The point of this script is that nothing about a watch is decided at build time:
you point it at footage, name a region, say what you want to be told, and it
reports what it saw — with the frames that decided it.

    uv run --extra lookout python scripts/lookout_replay.py clip.mp4 \
        --box 640,300,900,470 --condition gone --label "the pickup" --subject truck

Coordinates are in the clip's own pixel space. Use --arm-at to arm from a later
frame, and --arm-frame-out to look at the frame you just boxed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import cv2

from stopsign.lookout.manager import arm_watch
from stopsign.lookout.options import LookoutOptions
from stopsign.lookout.replay import run_replay
from stopsign.lookout.replay import write_reference_image
from stopsign.lookout.types import Condition


def parse_box(raw: str) -> tuple[float, float, float, float]:
    parts = [token for token in raw.replace(" ", "").split(",") if token]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("box must be x1,y1,x2,y2")
    try:
        x1, y1, x2, y2 = (float(value) for value in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"box values must be numbers: {exc}") from exc
    if x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError("box must have positive width and height")
    return (x1, y1, x2, y2)


def grab_frame(source: str, at_sec: float):
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"cannot open {source}")
    if at_sec > 0:
        capture.set(cv2.CAP_PROP_POS_MSEC, at_sec * 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise SystemExit(f"cannot read a frame at {at_sec}s from {source}")
    return frame


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a Lookout watch over recorded footage.")
    parser.add_argument("source", help="video file, RTSP URL, or HLS playlist")
    parser.add_argument("--box", type=parse_box, required=True, help="x1,y1,x2,y2 in the clip's pixel space")
    parser.add_argument(
        "--condition",
        choices=[item.value for item in Condition if item.selectable],
        default=Condition.GONE.value,
    )
    parser.add_argument("--label", default="target")
    parser.add_argument("--subject", default="", help="expected subject, e.g. car, truck, person (empty = anything)")
    parser.add_argument("--arm-at", type=float, default=0.0, help="seconds into the clip to arm from")
    parser.add_argument(
        "--start-at",
        type=float,
        default=None,
        help="seconds into the clip to begin watching (default: the arm time)",
    )
    parser.add_argument(
        "--armed",
        dest="armed_occupied",
        choices=["auto", "occupied", "empty"],
        default="auto",
        help="state of the region when armed (default: implied by the condition)",
    )
    parser.add_argument("--sample-hz", type=float, default=2.0, help="evaluation rate (the live default is 2)")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument(
        "--capture-base",
        type=float,
        default=0.0,
        help="epoch time of the clip's first frame, so evidence stamps match the footage",
    )
    parser.add_argument("--storage-root", default="/tmp/lookout-replay")
    parser.add_argument("--json", dest="json_out", default="")
    parser.add_argument("--arm-frame-out", default="", help="write the arming frame with the box drawn on it")
    parser.add_argument("--require-alert", action="store_true", help="exit non-zero when nothing alerting fired")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    condition = Condition(args.condition)
    armed_occupied = args.armed_occupied

    options = LookoutOptions.from_env()
    frame = grab_frame(args.source, args.arm_at)
    height, width = frame.shape[:2]

    watch = arm_watch(
        box=args.box,
        condition=condition,
        label=args.label,
        subject=args.subject,
        reference_size=(width, height),
        armed_occupied=None if armed_occupied == "auto" else armed_occupied == "occupied",
        created_by="replay",
        watch_id=f"rp{os.getpid() % 100000:05d}",
        now=args.arm_at,
    )
    watch.reference_image = write_reference_image(frame, watch.id, args.storage_root)

    if args.arm_frame_out:
        marked = frame.copy()
        x1, y1, x2, y2 = (int(value) for value in args.box)
        cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 214, 255), 3)
        cv2.imwrite(args.arm_frame_out, marked)
        if not args.quiet:
            print(f"arming frame written to {args.arm_frame_out} (reference crop: {watch.reference_image})")

    if not args.quiet:
        print(
            f"watch {watch.id}: {watch.kind.value} '{watch.label}' "
            f"condition={watch.condition.value} subject={watch.subject or 'any'} "
            f"box={args.box} armed_occupied={watch.armed_occupied}"
        )

    report = run_replay(
        args.source,
        [watch],
        options=options,
        sample_hz=args.sample_hz,
        max_seconds=args.max_seconds,
        storage_root=args.storage_root,
        start_sec=args.arm_at if args.start_at is None else args.start_at,
        time_base=args.capture_base,
        quiet=args.quiet,
    )

    if not args.quiet:
        print(
            f"\nreplayed {report.sampled_frames} sampled frames "
            f"({report.video_seconds:.1f}s of video at {report.fps:.1f}fps, "
            f"{report.frame_size[0]}x{report.frame_size[1]}) in {report.elapsed_sec:.1f}s"
        )
        print(f"transitions: {report.transitions() or 'none'}")
        for event in report.alerting:
            print(
                f"ALERT {event['label']}: {event['wording']} at {event['capture_ts']:.1f}s "
                f"(p={event['present_prob']:.2f}, evidence={event['evidence_fraction']:.0%}, "
                f"dir={event['evidence_dir']})"
            )
        for watch_id, status in report.final_states.items():
            fraction = f" [evidence {status['evidence_fraction']:.0%}]" if status["evidence_fraction"] else ""
            print(f"final state {watch_id}: {status['state']} — {status['reason']}{fraction}")
        for note in report.notes:
            print(f"note: {note}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2, default=str)
        if not args.quiet:
            print(f"report written to {args.json_out}")

    if args.require_alert and not report.alerting:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
