"""Pretrain a draft network against a frozen local target."""

import math
from pathlib import Path

import torch

from helper.arguments import arguments
from helper.checkpoints import load_checkpoint, restore_rng, rng_state, save_checkpoint
from helper.data import conversation_tokens, data_signature, read_rows, windows
from helper.objectives import anchor_plan, check_finite, draft_objective
from helper.runtime import setup


def main():
    args = arguments()
    saved = load_checkpoint(args.resume) if args.resume else None
    if saved is not None and saved.get("stage") != "pretrain":
        raise ValueError("Resume requires a pretraining checkpoint")
    if not saved and (Path(args.output) / "latest.pt").exists():
        raise FileExistsError(
            "Output already has a checkpoint; choose another output or use --resume"
        )
    target, tokenizer, draft = setup(args, saved)
    rows = [
        conversation_tokens(tokenizer, row, args.max_length)
        for row in read_rows(args.data)
    ]
    rows = [
        (ids, mask)
        for ids, mask in rows
        if any(mask[:-2] if args.method == "tree" else mask[1:])
    ]
    if not rows:
        raise ValueError("No assistant training tokens survive truncation")
    optimizer = torch.optim.AdamW(
        draft.parameters(),
        lr=args.lr,
        weight_decay=0.01 if args.method == "tree" else 0.0,
    )
    total_steps = args.epochs * math.ceil(
        math.ceil(len(rows) / args.batch_size) / args.accumulation
    )
    warmup = min(500, int(0.05 * total_steps))

    def schedule(step):
        if step < warmup:
            return step / max(1, warmup)
        phase = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * phase))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    settings = {
        name: getattr(args, name)
        for name in (
            "batch_size",
            "accumulation",
            "epochs",
            "max_length",
            "anchors",
            "lr",
            "seed",
            "dtype",
        )
    }
    settings["examples"] = len(rows)
    settings["data_signature"] = data_signature(rows)
    step = first_epoch = first_window = 0
    if saved:
        if settings != saved["settings"]:
            raise ValueError("Resume settings differ from checkpoint")
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        step, first_epoch, first_window = (
            saved[key] for key in ("step", "epoch", "next_window")
        )
        restore_rng(saved["rng"])
    del saved
    device = next(target.parameters()).device

    def save(epoch, cursor):
        save_checkpoint(
            args.output,
            step,
            {
                "stage": "pretrain",
                "spec": draft.spec,
                "draft": draft.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "epoch": epoch,
                "next_window": cursor,
                "settings": settings,
                "rng": rng_state(),
            },
        )

    if args.max_steps and step >= args.max_steps:
        return
    for epoch in range(first_epoch, args.epochs):
        for cursor, batches in enumerate(
            windows(len(rows), args.batch_size, args.accumulation, args.seed, epoch)
        ):
            if epoch == first_epoch and cursor < first_window:
                continue
            samples = []
            for batch in batches:
                for index in batch:
                    ids, mask = (
                        torch.tensor(value, device=device) for value in rows[index]
                    )
                    plan = (
                        anchor_plan(ids, mask, draft.spec, args.anchors)
                        if args.method == "block"
                        else None
                    )
                    samples.append((ids, mask, plan))
            denominator = (
                sum(float(plan[1].sum()) for _, _, plan in samples)
                if args.method == "block"
                else len(samples)
            )
            optimizer.zero_grad(set_to_none=True)
            draft.train()
            for ids, mask, plan in samples:
                numerator, count = draft_objective(
                    target, draft, ids, mask, args.loss_chunk_size, plan
                )
                loss = (
                    numerator / denominator
                    if args.method == "block"
                    else numerator / count / denominator
                )
                check_finite(loss)
                loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % args.save_every == 0:
                save(epoch, cursor + 1)
            if args.max_steps and step >= args.max_steps:
                save(epoch, cursor + 1)
                return
        first_window = 0
    save(args.epochs, 0)


if __name__ == "__main__":
    main()
