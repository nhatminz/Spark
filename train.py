"""Joint group-relative policy optimization and online draft distillation."""

from pathlib import Path

import torch
from common import (
    anchor_plan,
    arguments,
    check_finite,
    data_signature,
    draft_objective,
    load_checkpoint,
    policy_objective,
    policy_optimizer,
    precision,
    prompt_tokens,
    read_rows,
    restore_rng,
    rng_state,
    save_checkpoint,
    selected_logps,
    setup,
    windows,
)
from correction import Correction
from decode import generate
from model import attach_policy_adapter, policy_state, reference_policy, restore_policy


def reward_function():
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify

    configuration = [
        LatexExtractionConfig(
            normalization_config=NormalizationConfig(
                nits=False,
                malformed_operators=False,
                basic_latex=True,
                boxed="all",
                units=True,
            ),
            boxed_match_priority=0,
            try_extract_without_anchor=False,
        )
    ]

    def parse_gold(answer):
        gold = parse(
            answer,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if not gold:
            raise ValueError("Unparseable mathematical ground truth")
        return gold

    def evaluate(text, gold):
        answer = parse(
            text, extraction_mode="first_match", extraction_config=configuration
        )
        try:
            correct = float(verify(answer, gold))
        except (ValueError, TypeError, TimeoutError):
            correct = 0.0
        return correct + 0.2 * float(text.count("\n</think>\n") == 1)

    return parse_gold, evaluate


def update_draft(target, draft, optimizer, samples, args):
    eligible = []
    for ids, mask, _ in samples:
        if args.method == "tree":
            if mask[:-2].any():
                eligible.append((ids, mask, None))
        else:
            plan = anchor_plan(ids, mask, draft.spec, args.anchors)
            if plan is not None:
                eligible.append((ids, mask, plan))
    if not eligible:
        return
    denominator = (
        sum(float(plan[1].sum()) for _, _, plan in eligible)
        if args.method == "block"
        else len(eligible)
    )
    weight = (
        args.draft_weight
        if args.draft_weight is not None
        else (0.1 if args.method == "block" else 1.0)
    )
    optimizer.zero_grad(set_to_none=True)
    draft.train()
    for ids, mask, plan in eligible:
        numerator, count = draft_objective(
            target, draft, ids, mask, args.loss_chunk_size, plan
        )
        loss = (
            numerator / denominator
            if args.method == "block"
            else numerator / count / denominator
        )
        loss = loss * weight
        check_finite(loss)
        loss.backward()
    optimizer.step()


def main():
    args = arguments(training=True)
    saved = load_checkpoint(args.resume) if args.resume else None
    initial = saved or load_checkpoint(args.pretrained)
    if initial.get("stage") != ("train" if saved else "pretrain"):
        raise ValueError("Wrong checkpoint stage")
    if not saved and (Path(args.output) / "latest.pt").exists():
        raise FileExistsError(
            "Output already has a checkpoint; choose another output or use --resume"
        )
    target, tokenizer, draft = setup(args, initial)
    parse_gold, evaluate = reward_function()
    rows = [
        prompt_tokens(tokenizer, row, args.max_length) for row in read_rows(args.data)
    ]
    signature = data_signature(rows)
    rows = [(ids, parse_gold(answer)) for ids, answer in rows]
    if (
        args.max_length
        + args.max_new_tokens
        + max(draft.spec["block_size"], args.tree_depth + 1)
        > target.config.max_position_embeddings
    ):
        raise ValueError(
            "Prompt and response limits plus verification width exceed target context length"
        )
    device = next(target.parameters()).device
    adapted = args.policy_update == "adapter" or (
        args.policy_update == "native" and args.method == "tree"
    )
    reference = None
    if adapted:
        attach_policy_adapter(target, args.adapter_rank, args.adapter_alpha)
    else:
        from transformers import AutoModelForCausalLM

        reference = (
            AutoModelForCausalLM.from_pretrained(
                args.model,
                local_files_only=True,
                torch_dtype=target.dtype,
                attn_implementation="sdpa",
            )
            .to(args.reference_device or args.device)
            .eval()
            .requires_grad_(False)
        )
        target.requires_grad_(True)
    target.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    # Keep rollout and training probabilities identical while enabling activation
    # checkpointing, which requires training mode in the decoder backbone.
    for module in target.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if hasattr(module, "attention_dropout"):
            module.attention_dropout = 0.0
    decay = 0.01 if args.method == "tree" else 0.0
    optimizer = policy_optimizer(target, args.lr, decay)
    draft_optimizer = torch.optim.AdamW(
        draft.parameters(), lr=args.draft_lr, weight_decay=decay
    )
    correction = (
        Correction(
            target.get_output_embeddings().weight,
            args.correction_rank,
            args.correction_topk,
            args.correction_lr,
            args.projector_lr,
        )
        if args.correction
        else None
    )
    settings = {
        name: getattr(args, name)
        for name in (
            "method",
            "batch_size",
            "accumulation",
            "epochs",
            "max_length",
            "max_new_tokens",
            "responses",
            "temperature",
            "top_p",
            "beta",
            "epsilon",
            "lr",
            "draft_lr",
            "draft_weight",
            "policy_update",
            "adapter_rank",
            "adapter_alpha",
            "tree_width",
            "tree_depth",
            "tree_budget",
            "seed",
            "anchors",
            "correction",
            "correction_rank",
            "correction_topk",
            "correction_lr",
            "projector_lr",
            "dtype",
        )
    }
    settings["examples"] = len(rows)
    settings["data_signature"] = signature
    step = first_epoch = first_window = 0
    if saved:
        if settings != saved["settings"]:
            raise ValueError("Resume settings differ from checkpoint")
        restore_policy(target, saved["policy"], adapted)
        optimizer.load_state_dict(saved["optimizer"])
        draft_optimizer.load_state_dict(saved["draft_optimizer"])
        if correction:
            correction.load_state_dict(saved["correction"])
        step, first_epoch, first_window = (
            saved[key] for key in ("step", "epoch", "next_window")
        )
        restore_rng(saved["rng"])
    del saved, initial
    stop_ids = target.generation_config.eos_token_id
    stop_ids = (
        {stop_ids}
        if isinstance(stop_ids, int)
        else set(stop_ids or [tokenizer.eos_token_id])
    )

    def save(epoch, cursor):
        save_checkpoint(
            args.output,
            step,
            {
                "stage": "train",
                "spec": draft.spec,
                "draft": draft.state_dict(),
                "policy": policy_state(target, adapted),
                "optimizer": optimizer.state_dict(),
                "draft_optimizer": draft_optimizer.state_dict(),
                "correction": correction.state_dict() if correction else None,
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
            selected = []
            target.eval()
            for batch in batches:
                if correction:
                    correction.reset()
                samples = []
                for index in batch:
                    prompt, gold = rows[index]
                    completions = [
                        generate(target, draft, prompt, stop_ids, args, correction)
                        for _ in range(args.responses)
                    ]
                    rewards = torch.tensor(
                        [
                            evaluate(
                                tokenizer.decode(tokens, skip_special_tokens=True), gold
                            )
                            for tokens in completions
                        ],
                        device=device,
                    )
                    deviation = rewards.std(unbiased=False)
                    advantages = (
                        (rewards - rewards.mean()) / deviation
                        if deviation > 0
                        else torch.zeros_like(rewards)
                    )
                    for tokens, advantage in zip(completions, advantages):
                        ids = torch.tensor(prompt + tokens, device=device)
                        mask = torch.arange(ids.numel(), device=device) >= len(prompt)
                        samples.append((ids, mask, advantage))
                        if deviation > 0:
                            selected.append((ids.cpu(), mask.cpu(), advantage.detach()))
                # The policy stays fixed throughout the accumulation window.
                # Draft updates use every response, including zero-variance groups.
                update_draft(target, draft, draft_optimizer, samples, args)
                if correction:
                    correction.step()
            optimizer.zero_grad(set_to_none=True)
            for ids, mask, advantage in selected:
                ids, mask = ids.to(device), mask.to(device)[1:]
                target.eval()
                with torch.no_grad():
                    if adapted:
                        with reference_policy(target), precision(target):
                            reference_logps = selected_logps(target, ids)
                    else:
                        reference_device = next(reference.parameters()).device
                        with precision(reference):
                            reference_logps = selected_logps(
                                reference, ids.to(reference_device)
                            ).to(device)
                target.train()
                with precision(target):
                    logps = selected_logps(target, ids)
                    loss = policy_objective(
                        logps, reference_logps, advantage, mask, args.beta, args.epsilon
                    ) / len(selected)
                check_finite(loss)
                loss.backward()
                if hasattr(optimizer, "accumulate"):
                    optimizer.accumulate()
            if selected:
                optimizer.step()
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
