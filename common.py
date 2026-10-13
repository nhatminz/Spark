"""Local input, objectives, and training state shared by both entry points."""

import argparse
import hashlib
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from model import make_draft, specification
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging

logging.set_verbosity_error()
logging.disable_progress_bar()


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


def read_rows(path):
    path = Path(path)
    if path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif path.suffix == ".parquet":
        from pyarrow import parquet

        rows = parquet.read_table(path).to_pylist()
    else:
        raise ValueError("Data must be JSON, JSONL, or Parquet")
    if (
        not isinstance(rows, list)
        or not rows
        or not all(isinstance(row, dict) for row in rows)
    ):
        raise ValueError("Data must contain a nonempty list of objects")
    return rows


def data_signature(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, ensure_ascii=True, separators=(",", ":")).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def messages(row):
    turns = row.get("messages", row.get("conversations"))
    if not isinstance(turns, list) or not turns:
        raise ValueError("Pretraining requires messages or conversations")
    result = []
    for turn in turns:
        role = turn.get("role", turn.get("from"))
        role = {"human": "user", "gpt": "assistant", "ASSISTANT": "assistant"}.get(
            role, role
        )
        content = turn.get("content", turn.get("value"))
        if role not in {"system", "user", "assistant", "tool"} or not isinstance(
            content, str
        ):
            raise ValueError("Invalid conversation role or content")
        result.append({"role": role, "content": content})
    if result[0]["role"] != "system":
        result.insert(0, {"role": "system", "content": "You are a helpful assistant."})
    return result


def conversation_tokens(tokenizer, row, limit):
    turns = messages(row)
    ids, mask = [], []
    for index, turn in enumerate(turns):
        current = tokenizer.apply_chat_template(
            turns[: index + 1], tokenize=True, add_generation_prompt=False
        )
        if current[: len(ids)] != ids:
            raise ValueError(
                "Chat template must preserve token prefixes when adding turns"
            )
        mask.extend([int(turn["role"] == "assistant")] * (len(current) - len(ids)))
        ids = current
    return ids[:limit], mask[:limit]


def prompt_tokens(tokenizer, row, limit):
    prompt = row.get("messages", row.get("prompt"))
    if prompt is None:
        question = row.get("question")
        if not isinstance(question, str):
            raise ValueError("Training rows require question, prompt, or messages")
        prompt = [
            {"role": "system", "content": "You are a math problem assistant."},
            {
                "role": "user",
                "content": (
                    "Below is an instruction that describes a task, paired with an input that provides further context.\n"
                    "            Write a response that appropriately completes the request.\n"
                    "            Your response should include your thought process enclosed within <think></think> tags\n"
                    "            and the final answer enclosed within <answer></answer> tags (Just put a number between the tags).\n\n"
                    "            ### Instruction:\n"
                    + question
                    + "\nPlease reason step by step, and put your final answer within \\boxed{}"
                ),
            },
        ]
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    if (
        not isinstance(prompt, list)
        or not prompt
        or prompt[-1].get("role") == "assistant"
    ):
        raise ValueError("Prompt must end before the assistant response")
    ids = tokenizer.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True
    )
    if not ids or len(ids) > limit:
        raise ValueError("Prompt is empty or exceeds max-length; adjust max-length")
    answer = row.get("answer")
    if answer is None and isinstance(row.get("reward_model"), dict):
        answer = row["reward_model"].get("ground_truth")
    if answer is None:
        raise ValueError("Training rows require answer or reward_model.ground_truth")
    answer = str(answer).split("####")[-1].strip()
    if "\\boxed{" not in answer:
        answer = "\\boxed{" + answer + "}"
    return ids, answer


def windows(length, batch_size, accumulation, seed, epoch):
    order = list(range(length))
    random.Random(seed + epoch).shuffle(order)
    batches = [
        order[start : start + batch_size] for start in range(0, length, batch_size)
    ]
    return [
        batches[start : start + accumulation]
        for start in range(0, len(batches), accumulation)
    ]


def soft_cross_entropy(student, teacher, head, chunk_size):
    weight = head.weight.detach()
    bias = None if head.bias is None else head.bias.detach()

    def evaluate(x, y):
        with torch.no_grad():
            probabilities = (
                F.linear(y.to(weight.dtype), weight, bias).float().softmax(-1)
            )
        logits = F.linear(x.to(weight.dtype), weight, bias).float()
        return -(probabilities * logits.log_softmax(-1)).sum(-1)

    flat_x, flat_y = (
        student.reshape(-1, student.shape[-1]),
        teacher.reshape(-1, teacher.shape[-1]),
    )
    losses = []
    for start in range(0, flat_x.shape[0], chunk_size):
        losses.append(
            checkpoint(
                evaluate,
                flat_x[start : start + chunk_size],
                flat_y[start : start + chunk_size],
                use_reentrant=False,
            )
        )
    return torch.cat(losses).reshape(student.shape[:-1])


def anchor_plan(ids, mask, spec, count):
    candidates = torch.nonzero(mask[1:].bool(), as_tuple=True)[0].cpu()
    if not candidates.numel():
        return None
    generator = torch.Generator().manual_seed(int(ids.sum()) % (2**63 - 1))
    picks = (
        torch.randperm(candidates.numel(), generator=generator)[:count]
        if candidates.numel() >= count
        else torch.randint(candidates.numel(), (count,), generator=generator)
    )
    anchors = candidates[picks].to(ids.device)
    slots = torch.arange(spec["block_size"] - 1, device=ids.device)
    positions = anchors[:, None] + 1 + slots
    live = (positions < ids.numel()) & mask[positions.clamp_max(ids.numel() - 1)].bool()
    live = live.int().cumprod(-1).bool()
    size = spec["block_size"]
    decay = max(1.0, 4.0 + (size - 8) / 2 if size <= 10 else 5.0 + (size - 10) / 3)
    weights = torch.exp(-slots.float() / decay) * live
    return anchors, weights


def draft_objective(target, draft, ids, mask, chunk_size, plan=None):
    spec = draft.spec
    with torch.no_grad(), precision(target):
        output, features = context_forward(target, ids[None], spec, use_cache=False)
    with precision(target):
        if spec["method"] == "tree":
            features = output.last_hidden_state[:, :-1].detach()
            embeddings = target.get_input_embeddings()(ids[None, 1:]).detach()
            predicted, hidden, _ = draft(features, embeddings)
            valid = mask[:-2].bool()
            if not valid.any():
                return None
            feature_loss = F.smooth_l1_loss(
                predicted[0, :-1].float(), features[0, 1:].float(), reduction="none"
            ).mean(-1)
            ce = soft_cross_entropy(
                hidden[0, :-1],
                features[0, 1:],
                target.get_output_embeddings(),
                chunk_size,
            )
            return ((2 * feature_loss + 0.1 * ce) * valid).sum(), valid.sum()
        if plan is None:
            return None
        anchors, weights = plan
        features = features.detach()
        blocks = ids.new_full((1, anchors.numel(), spec["block_size"]), spec["mask_id"])
        blocks[0, :, 0] = ids[anchors]
        embeddings = target.get_input_embeddings()(blocks).detach()
        hidden = draft(features, embeddings, anchors[None])[0, :, 1:]
        positions = anchors[:, None] + torch.arange(
            spec["block_size"] - 1, device=ids.device
        )
        teacher = output.last_hidden_state[
            0, positions.clamp_max(ids.numel() - 1)
        ].detach()
        ce = soft_cross_entropy(
            hidden, teacher, target.get_output_embeddings(), chunk_size
        )
        return (ce * weights).sum(), weights.sum()


def selected_logps(target, ids):
    output = backbone(target, ids[None], use_cache=False).last_hidden_state[0, :-1]
    head = target.get_output_embeddings()
    result = []

    # Checkpoint the vocabulary projection so backward does not retain S*V logits.
    def evaluate(hidden, labels):
        logits = head(hidden).float()
        return logits.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)

    for start in range(0, output.shape[0], 32):
        result.append(
            checkpoint(
                evaluate,
                output[start : start + 32],
                ids[start + 1 : start + 33],
                use_reentrant=False,
            )
        )
    return torch.cat(result)


def policy_objective(logps, reference, advantage, mask, beta, epsilon):
    ratio = (logps - logps.detach()).exp()
    clipped = ratio.clamp(1 - epsilon, 1 + epsilon)
    delta = reference.detach() - logps
    terms = -torch.minimum(ratio * advantage, clipped * advantage) + beta * (
        delta.exp() - delta - 1
    )
    return (terms * mask).sum() / mask.sum()


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def save_checkpoint(output, step, payload):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"step_{step}.pt"
    temporary = output / ".checkpoint.tmp"
    torch.save(payload, temporary)
    temporary.replace(path)
    link = output / ".latest.tmp"
    link.unlink(missing_ok=True)
    link.symlink_to(path.name)
    link.replace(output / "latest.pt")


def rng_state():
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    torch.set_rng_state(state["cpu"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def check_finite(loss):
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite objective; refusing to update parameters")


class MasterOptimizer:
    """FP32 weights, moments, and accumulated gradients for a BF16 policy."""

    def __init__(self, parameters, learning_rate, weight_decay):
        self.parameters = parameters
        self.master = [
            torch.nn.Parameter(parameter.detach().float().clone())
            for parameter in parameters
        ]
        self.optimizer = torch.optim.AdamW(
            self.master, lr=learning_rate, weight_decay=weight_decay
        )

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for parameter in self.parameters:
            parameter.grad = None

    @torch.no_grad()
    def accumulate(self):
        for parameter, master in zip(self.parameters, self.master):
            if parameter.grad is not None:
                gradient = parameter.grad.float()
                if master.grad is None:
                    master.grad = gradient
                else:
                    master.grad.add_(gradient)
                parameter.grad = None

    @torch.no_grad()
    def step(self):
        self.accumulate()
        self.optimizer.step()
        for parameter, master in zip(self.parameters, self.master):
            parameter.copy_(master)

    def state_dict(self):
        state = self.optimizer.state_dict()
        state["master_parameters"] = [parameter.detach() for parameter in self.master]
        return state

    def load_state_dict(self, state):
        state = dict(state)
        weights = state.pop("master_parameters")
        if len(weights) != len(self.master):
            raise ValueError("Master parameter count mismatch")
        with torch.no_grad():
            for parameter, master, value in zip(self.parameters, self.master, weights):
                master.copy_(value)
                parameter.copy_(master)
        self.optimizer.load_state_dict(state)


def policy_optimizer(target, learning_rate, weight_decay):
    parameters = [
        parameter for parameter in target.parameters() if parameter.requires_grad
    ]
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        return MasterOptimizer(parameters, learning_rate, weight_decay)
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
