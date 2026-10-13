"""Command-line arguments for pretraining and policy training."""

import argparse
import math


def arguments(training=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("tree", "block"), required=True)
    parser.add_argument("--model", required=True, help="Local target model directory")
    parser.add_argument(
        "--data", required=True, help="Local JSON, JSONL, or Parquet file"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--resume", default=None, help="Checkpoint produced by this entry point"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Examples/prompts per batch"
    )
    parser.add_argument(
        "--accumulation", type=int, default=4, help="Batches per iteration"
    )
    parser.add_argument("--epochs", type=int, default=1 if training else 5)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--max-steps", type=int, default=0, help="0 processes all epochs"
    )
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Anchor plus masked slots (default: 16)",
    )
    parser.add_argument(
        "--mask-id", type=int, default=None, help="Unused target embedding row"
    )
    parser.add_argument("--anchors", type=int, default=16)
    parser.add_argument("--loss-chunk-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-6 if training else 5e-5)
    if training:
        parser.add_argument("--pretrained", help="Draft checkpoint from pretrain.py")
        parser.add_argument("--responses", type=int, default=8)
        parser.add_argument("--max-new-tokens", type=int, default=2048)
        parser.add_argument("--temperature", type=float, default=1.0)
        parser.add_argument("--top-p", type=float, default=0.95)
        parser.add_argument("--beta", type=float, default=0.04)
        parser.add_argument("--epsilon", type=float, default=0.1)
        parser.add_argument("--draft-lr", type=float, default=1e-4)
        parser.add_argument("--draft-weight", type=float, default=None)
        parser.add_argument(
            "--policy-update", choices=("native", "adapter", "full"), default="native"
        )
        parser.add_argument("--adapter-rank", type=int, default=64)
        parser.add_argument("--adapter-alpha", type=float, default=32.0)
        parser.add_argument(
            "--reference-device",
            default=None,
            help="Full reference defaults to policy device",
        )
        parser.add_argument("--tree-width", type=int, default=8)
        parser.add_argument("--tree-depth", type=int, default=5)
        parser.add_argument(
            "--tree-budget", type=int, default=160, help="Includes verification root"
        )
        parser.add_argument("--correction", action="store_true")
        parser.add_argument("--correction-rank", type=int, default=8)
        parser.add_argument("--correction-topk", type=int, default=16)
        parser.add_argument("--correction-lr", type=float, default=0.01)
        parser.add_argument("--projector-lr", type=float, default=1e-4)
    args = parser.parse_args()
    positive = (
        "batch_size",
        "accumulation",
        "epochs",
        "save_every",
        "anchors",
        "loss_chunk_size",
    )
    if (
        any(getattr(args, name) < 1 for name in positive)
        or args.max_length < 3
        or args.max_steps < 0
    ):
        parser.error(
            "Training counts must be positive; max-length >= 3; max-steps >= 0"
        )
    if (
        not math.isfinite(args.lr)
        or args.lr <= 0
        or (args.block_size is not None and not 2 <= args.block_size <= 64)
    ):
        parser.error("Require lr > 0 and block-size in [2, 64]")
    if args.layers is not None and args.layers < 1:
        parser.error("layers must be positive")
    if training:
        if not args.resume and not args.pretrained:
            parser.error("train requires --pretrained or --resume")
        if (
            min(
                args.responses,
                args.tree_width,
                args.tree_depth,
                args.tree_budget,
                args.adapter_rank,
                args.correction_rank,
                args.correction_topk,
                args.max_new_tokens,
            )
            < 1
            or args.responses < 2
        ):
            parser.error(
                "Positive decoding counts and at least two responses are required"
            )
        if args.temperature < 0 or not 0 < args.top_p <= 1 or not 0 <= args.epsilon < 1:
            parser.error("Invalid temperature, top-p, or epsilon")
        rates = (
            args.beta,
            args.draft_lr,
            args.adapter_alpha,
            args.correction_lr,
            args.projector_lr,
        )
        if (
            any(not math.isfinite(value) or value < 0 for value in rates)
            or args.draft_lr == 0
        ):
            parser.error(
                "Learning rates and coefficients must be finite and nonnegative"
            )
        if args.draft_weight is not None and (
            not math.isfinite(args.draft_weight) or args.draft_weight <= 0
        ):
            parser.error("draft-weight must be positive")
        if not math.isfinite(args.temperature):
            parser.error("temperature must be finite")
    return args
