"""Publish one checkpoint through vLLM's NCCL weight-transfer engine."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hf_checkpoint_nccl_publisher import (
    CheckpointManifest,
    NcclCheckpointPublisher,
    validate_manifest_transfer_order,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--draft-model")
    parser.add_argument("--draft-revision")
    parser.add_argument("--draft-checkpoint-path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--update-bucket-size-mb", type=int, default=512)
    parser.add_argument(
        "--expert-tensor-order",
        choices=("natural", "lexical", "physical"),
        default="natural",
        help=(
            "Order tensors within each complete expert layer. 'lexical' "
            "matches checkpoints written in lexical name order; 'physical' "
            "uses each tensor's actual safetensors file and byte offset."
        ),
    )
    parser.add_argument(
        "--direct-file-expert-h2d",
        action="store_true",
        help=(
            "Copy each physically contiguous expert safetensors span "
            "directly from its mmap into one GPU allocation."
        ),
    )
    parser.add_argument(
        "--nonexpert-tensor-order",
        choices=("natural", "physical"),
        default="natural",
        help="Order nonexpert tensors by name or safetensors file offset.",
    )
    parser.add_argument(
        "--direct-file-h2d",
        action="store_true",
        help="Copy physically adjacent tensors in any bucket as one GPU span.",
    )
    parser.add_argument(
        "--large-tensor-chunk-mb",
        type=int,
        default=0,
        help=(
            "Stream tensors larger than this size through a bounded host buffer "
            "before transfer (0 disables streaming)."
        ),
    )
    parser.add_argument(
        "--large-tensor-read-workers",
        type=int,
        default=1,
        help="Concurrent file readers for streamed large tensors (default: 1).",
    )
    parser.add_argument(
        "--prefetch-max-bucket-mb",
        type=int,
        default=0,
        help=(
            "Overlap the next bucket's source materialization with NCCL/receiver "
            "work when both buckets fit this per-bucket limit; 0 disables it. "
            "Reserve GPU memory for two eligible buckets."
        ),
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--enable-mtp",
        action="store_true",
        help="Send the same canonical checkpoint to the draft model after main.",
    )
    parser.add_argument(
        "--enable-draft-update",
        action="store_true",
        help="Send an independent checkpoint to the speculative draft model.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.enable_mtp and args.enable_draft_update:
        parser.error("--enable-mtp and --enable-draft-update are mutually exclusive")
    if args.large_tensor_chunk_mb < 0:
        parser.error("large tensor chunk size must be nonnegative")
    if args.large_tensor_read_workers < 1:
        parser.error("large tensor read workers must be positive")
    if args.prefetch_max_bucket_mb < 0:
        parser.error("prefetch max bucket size must be nonnegative")
    if args.enable_draft_update and not (
        args.draft_model
        and args.draft_revision
        and args.draft_checkpoint_path
    ):
        parser.error(
            "--enable-draft-update requires --draft-model, --draft-revision, "
            "and --draft-checkpoint-path"
        )

    run_started = time.perf_counter()
    manifest_started = time.perf_counter()
    manifest = CheckpointManifest.load(
        model=args.model,
        revision=args.revision,
        checkpoint_path=args.checkpoint_path,
        expert_tensor_order=args.expert_tensor_order,
        nonexpert_tensor_order=args.nonexpert_tensor_order,
    )
    draft_manifest = None
    if args.enable_draft_update:
        draft_manifest = CheckpointManifest.load(
            model=args.draft_model,
            revision=args.draft_revision,
            checkpoint_path=args.draft_checkpoint_path,
            expert_tensor_order=args.expert_tensor_order,
            nonexpert_tensor_order=args.nonexpert_tensor_order,
        )
    validate_manifest_transfer_order(
        manifest, args.update_bucket_size_mb * 1024**2
    )
    if draft_manifest is not None:
        validate_manifest_transfer_order(
            draft_manifest, args.update_bucket_size_mb * 1024**2
        )
    manifest_seconds = time.perf_counter() - manifest_started
    publisher = NcclCheckpointPublisher(
        base_url=args.base_url,
        manifest=manifest,
        device=args.device,
        update_bucket_size_bytes=args.update_bucket_size_mb * 1024**2,
        direct_file_expert_h2d=args.direct_file_expert_h2d,
        direct_file_h2d=args.direct_file_h2d,
        large_tensor_chunk_bytes=args.large_tensor_chunk_mb * 1024**2,
        large_tensor_read_workers=args.large_tensor_read_workers,
        prefetch_max_bucket_bytes=args.prefetch_max_bucket_mb * 1024**2,
        timeout_seconds=args.timeout,
    )
    try:
        init_started = time.perf_counter()
        initialization = publisher.initialize()
        init_seconds = time.perf_counter() - init_started
        update_started = time.perf_counter()
        updates = [publisher.publish_update(target="main")]
        if args.enable_mtp:
            updates.append(publisher.publish_update(target="draft"))
        elif draft_manifest is not None:
            updates.append(
                publisher.publish_update(target="draft", manifest=draft_manifest)
            )
        update_seconds = time.perf_counter() - update_started
    finally:
        publisher.shutdown()

    result = {
        "checkpoint": manifest.summary(),
        "initialization": initialization,
        "mtp_enabled": args.enable_mtp,
        "independent_draft_enabled": draft_manifest is not None,
        "draft_checkpoint": (
            draft_manifest.summary() if draft_manifest is not None else None
        ),
        "update": updates[0],
        "updates": updates,
        "timings": {
            "manifest_seconds": manifest_seconds,
            "initialize_seconds": init_seconds,
            "update_seconds": update_seconds,
            "total_seconds": time.perf_counter() - run_started,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"send_weights_completed": True}))
if __name__ == "__main__":
    main()
