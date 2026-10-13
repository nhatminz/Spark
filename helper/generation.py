"""Tree token matching and parallel block rejection sampling."""

import torch
from torch.nn import functional as F
from transformers import DynamicCache

from .runtime import backbone, context_forward, precision


def probabilities(logits, temperature, top_p):
    if temperature == 0:
        return F.one_hot(logits.argmax(-1), logits.shape[-1]).float()
    result = (logits.float() / temperature).softmax(-1)
    if top_p < 1:
        ordered, indices = result.sort(descending=True, dim=-1)
        ordered *= ordered.cumsum(-1) - ordered < top_p
        result = torch.zeros_like(result).scatter(-1, indices, ordered)
        result /= result.sum(-1, keepdim=True)
    return result


def sample(p):
    return torch.multinomial(p.reshape(-1, p.shape[-1]), 1).reshape(p.shape[:-1])


def head_logits(target, hidden):
    return target.get_output_embeddings()(hidden.to(target.dtype))


def compact_cache(cache, indices):
    """Build a cache containing only committed prefix/path rows."""
    result = DynamicCache()
    for index in range(len(cache)):
        if hasattr(cache, "layers"):
            keys, values = cache.layers[index].keys, cache.layers[index].values
        else:
            keys, values = cache[index]
        result.update(
            keys.index_select(2, indices), values.index_select(2, indices), index
        )
    return result


def choose_cache(cache, indices):
    return [
        (key.index_select(0, indices), value.index_select(0, indices))
        for key, value in cache
    ]


def grow_tree(target, draft, ids, features, anchor, args, correction):
    shifted = torch.cat((ids[1:], anchor.reshape(1)))[None]
    predicted, hidden, cache = draft(features, target.get_input_embeddings()(shifted))
    next_features, current_hidden = predicted[:, -1:], hidden[:, -1:]
    contexts = {0: current_hidden[0, 0]}
    # Each record stores token, parent, depth, and cumulative proposal probability.
    records, frontier, scores = [], [0], torch.ones(1, device=ids.device)
    width = min(args.tree_width, target.config.vocab_size)
    for depth in range(1, args.tree_depth + 1):
        logits = head_logits(target, current_hidden[:, -1])
        if correction is not None:
            logits = correction.apply(current_hidden[:, -1], logits)
        values, tokens = logits.float().softmax(-1).topk(width, -1)
        confidence = scores[:, None] * values
        offset = len(records) + 1
        flat_tokens = tokens.reshape(-1).tolist()
        flat_scores = confidence.reshape(-1).tolist()
        for index, token in enumerate(flat_tokens):
            records.append((token, frontier[index // width], depth, flat_scores[index]))
        if depth == args.tree_depth:
            break
        selected = confidence.flatten().topk(min(width, confidence.numel())).indices
        parents = selected // width
        cache = choose_cache(cache, parents)
        inputs = tokens.flatten()[selected, None]
        previous = next_features.index_select(0, parents)
        predicted, hidden, cache = draft(
            previous, target.get_input_embeddings()(inputs), cache=cache
        )
        next_features, current_hidden = predicted, hidden
        scores = confidence.flatten()[selected]
        frontier = (selected + offset).tolist()
        contexts.update({node: hidden[index, 0] for index, node in enumerate(frontier)})
    # Cumulative confidence decreases along every path. Break ties by creation order
    # to retain ancestors; then restore packed creation order for token matching.
    chosen = sorted(
        sorted(
            range(1, len(records) + 1), key=lambda node: (-records[node - 1][3], node)
        )[: max(0, args.tree_budget - 1)]
    )
    mapping = {0: 0, **{node: index + 1 for index, node in enumerate(chosen)}}
    if any(records[node - 1][1] not in mapping for node in chosen):
        raise RuntimeError("Selected tree is not parent closed")
    tokens = [int(anchor)] + [records[node - 1][0] for node in chosen]
    parents = [-1] + [mapping[records[node - 1][1]] for node in chosen]
    depths = [0] + [records[node - 1][2] for node in chosen]
    contexts = {
        mapping[node]: value for node, value in contexts.items() if node in mapping
    }
    return tokens, parents, depths, contexts


def tree_mask(parents, prefix, dtype, device):
    count = len(parents)
    mask = torch.full(
        (count, prefix + count), torch.finfo(dtype).min, dtype=dtype, device=device
    )
    mask[:, :prefix] = 0
    for row in range(count):
        node = row
        while node >= 0:
            mask[row, prefix + node] = 0
            node = parents[node]
    return mask[None, None]


@torch.no_grad()
def generate(target, draft, prompt, stop_ids, args, correction=None):
    device = next(target.parameters()).device
    ids = torch.tensor(prompt, device=device, dtype=torch.long)
    draft.eval()
    with precision(target):
        initial, taps = context_forward(target, ids[None], draft.spec, use_cache=True)
        cache = initial.past_key_values
        features = initial.last_hidden_state
        anchor = sample(
            probabilities(
                head_logits(target, features[:, -1]), args.temperature, args.top_p
            )
        )[0]
        emitted = [int(anchor)]
        while len(emitted) < args.max_new_tokens and emitted[-1] not in stop_ids:
            prefix = ids.numel()
            if draft.spec["method"] == "tree":
                tokens, parents, depths, contexts = grow_tree(
                    target, draft, ids, features, anchor, args, correction
                )
                packed = torch.tensor(tokens, device=device)[None]
                positions = torch.tensor(depths, device=device)[None] + prefix
                output = backbone(
                    target,
                    packed,
                    attention_mask=tree_mask(parents, prefix, target.dtype, device),
                    position_ids=positions,
                    past_key_values=cache,
                    use_cache=True,
                )
                p = probabilities(
                    head_logits(target, output.last_hidden_state[0]),
                    args.temperature,
                    args.top_p,
                )
                decisions = sample(p).tolist()
                current, committed, additional, feedback_rows = 0, [], [], []
                while True:
                    committed.append(current)
                    token = decisions[current]
                    additional.append(token)
                    if current in contexts:
                        feedback_rows.append(current)
                    if (
                        token in stop_ids
                        or len(emitted) + len(additional) >= args.max_new_tokens
                    ):
                        break
                    match = next(
                        (
                            index
                            for index in range(1, len(tokens))
                            if parents[index] == current and tokens[index] == token
                        ),
                        None,
                    )
                    if match is None:
                        break
                    current = match
                if correction is not None and feedback_rows:
                    hidden = torch.stack([contexts[row] for row in feedback_rows])
                    correction.feedback(
                        hidden, head_logits(target, hidden), p[feedback_rows]
                    )
            else:
                size = draft.spec["block_size"]
                # Verifier can compute unused future slots; only the emitted prefix is committed.
                blocks = ids.new_full((1, 1, size), draft.spec["mask_id"])
                blocks[0, 0, 0] = anchor
                hidden = draft(
                    taps,
                    target.get_input_embeddings()(blocks),
                    ids.new_tensor([[prefix]]),
                )[0, 0, 1:]
                raw = head_logits(target, hidden)
                corrected = raw if correction is None else correction.apply(hidden, raw)
                proposals = corrected.argmax(-1)
                packed = torch.cat((anchor.reshape(1), proposals))[None]
                output, new_taps = context_forward(
                    target, packed, draft.spec, past_key_values=cache, use_cache=True
                )
                p = probabilities(
                    head_logits(target, output.last_hidden_state[0]),
                    args.temperature,
                    args.top_p,
                )
                additional, committed, visited = [], [0], []
                for slot, proposed in enumerate(proposals.tolist()):
                    visited.append(slot)
                    accepted = bool(torch.rand((), device=device) < p[slot, proposed])
                    if accepted:
                        additional.append(proposed)
                    else:
                        residual = p[slot].clone()
                        residual[proposed] = 0
                        if residual.sum() <= 0:
                            raise FloatingPointError("Rejection residual has zero mass")
                        additional.append(int(sample(residual / residual.sum())))
                    if (
                        not accepted
                        or additional[-1] in stop_ids
                        or len(emitted) + len(additional) >= args.max_new_tokens
                    ):
                        break
                    committed.append(slot + 1)
                else:
                    additional.append(int(sample(p[-1])))
                # Last emitted token is the unprocessed anchor for the next round.
                committed = committed[: len(additional)]
                if correction is not None and visited:
                    correction.feedback(hidden[visited], raw[visited], p[visited])
            rows = torch.tensor(committed, device=device)
            selected = torch.cat((torch.arange(prefix, device=device), prefix + rows))
            cache = compact_cache(output.past_key_values, selected)
            ids = torch.cat((ids, packed[0, rows]))
            features = torch.cat((features, output.last_hidden_state[:, rows]), 1)
            if taps is not None:
                taps = torch.cat((taps, new_taps[:, rows]), 1)
            emitted.extend(additional)
            anchor = ids.new_tensor(additional[-1])
    return emitted[: args.max_new_tokens]
