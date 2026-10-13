"""Local target loading, precision, and decoder context features."""

import os
import random
from contextlib import nullcontext

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging

from models.draft import make_draft, specification

logging.set_verbosity_error()
logging.disable_progress_bar()


def precision(target):
    device = next(target.parameters()).device
    dtype = next(target.parameters()).dtype
    return (
        torch.autocast(device.type, dtype=dtype)
        if dtype == torch.bfloat16
        else nullcontext()
    )


def backbone(target, ids, **kwargs):
    return target.model(input_ids=ids, **kwargs)


def context_forward(target, ids, spec, **kwargs):
    captured, handles = {}, []
    if spec["method"] == "block":
        for index in spec["taps"]:

            def capture(module, inputs, output, key=index):
                captured[key] = output[0] if isinstance(output, tuple) else output

            handles.append(target.model.layers[index].register_forward_hook(capture))
    try:
        output = backbone(target, ids, output_hidden_states=False, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
    taps = (
        torch.cat([captured[index] for index in spec["taps"]], -1) if captured else None
    )
    return output, taps


def setup(args, saved=None):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use one process and one selected device per run")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.dtype == "bf16" and not str(args.device).startswith("cuda"):
        raise ValueError("Use --dtype fp32 for CPU runs")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    target = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        )
        .to(args.device)
        .eval()
        .requires_grad_(False)
    )
    if not hasattr(target, "model") or not hasattr(target.model, "layers"):
        raise ValueError("Target must expose a decoder backbone with a layers list")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.eos_token_id is None or not tokenizer.chat_template:
        raise ValueError("A chat template and an EOS token are required")
    if saved is None:
        layers = args.layers or (1 if args.method == "tree" else 3)
        mask_id = args.mask_id
        if args.method == "block" and mask_id is None:
            used = set(tokenizer.get_vocab().values())
            mask_id = next(
                (
                    index
                    for index in range(max(used) + 1, target.config.vocab_size)
                    if index not in used
                ),
                None,
            )
            if mask_id is None:
                raise ValueError(
                    "No unused embedding row; supply --mask-id after checking the vocabulary"
                )
        spec = specification(
            target.config, args.method, layers, args.block_size or 16, mask_id
        )
    else:
        spec = saved["spec"]
        expected = specification(
            target.config,
            args.method,
            spec["layers"],
            spec["block_size"],
            spec["mask_id"],
        )
        if spec != expected:
            raise ValueError("Checkpoint architecture differs from the target/method")
        for name, value in (
            ("layers", args.layers),
            ("mask_id", args.mask_id),
            ("block_size", args.block_size),
        ):
            if value is not None and value != spec[name]:
                raise ValueError(f"Explicit {name} differs from checkpoint")
    if spec["method"] == "block":
        mask_id = spec["mask_id"]
        if (
            mask_id in tokenizer.get_vocab().values()
            or not 0 <= mask_id < target.config.vocab_size
        ):
            raise ValueError("MASK must be an unused target embedding row")
    draft = make_draft(spec, args.device)
    if saved is not None:
        draft.load_state_dict(saved["draft"])
    return target, tokenizer, draft
