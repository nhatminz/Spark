"""Draft distillation and sequence-level group-relative objectives."""

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .runtime import backbone, context_forward, precision


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


def check_finite(loss):
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite objective; refusing to update parameters")


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
