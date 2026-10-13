"""Draft networks and optional low-rank policy updates."""

from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


class Norm(nn.Module):
    def __init__(self, width, epsilon):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.epsilon)
        return (y * self.weight).to(x.dtype)


class FeedForward(nn.Module):
    def __init__(self, width, intermediate):
        super().__init__()
        self.gate = nn.Linear(width, intermediate, bias=False)
        self.up = nn.Linear(width, intermediate, bias=False)
        self.down = nn.Linear(intermediate, width, bias=False)

    def forward(self, x, gate_input=None):
        gate_input = x if gate_input is None else gate_input
        return self.down(F.silu(self.gate(gate_input)) * self.up(x))


def rotate(x, positions, theta):
    width = x.shape[-1]
    inverse = theta ** (-torch.arange(0, width, 2, device=x.device).float() / width)
    angles = positions.float()[..., None] * inverse
    angles = torch.cat((angles, angles), -1)[:, None]
    left, right = x.chunk(2, dim=-1)
    return x * angles.cos().to(x.dtype) + torch.cat(
        (-right, left), -1
    ) * angles.sin().to(x.dtype)


class Attention(nn.Module):
    def __init__(self, spec, normalized=False):
        super().__init__()
        width, self.heads, self.kv_heads, self.dim = (
            spec[key] for key in ("width", "heads", "kv_heads", "head_dim")
        )
        self.theta = spec["theta"]
        self.query = nn.Linear(width, self.heads * self.dim, bias=False)
        self.key = nn.Linear(width, self.kv_heads * self.dim, bias=False)
        self.value = nn.Linear(width, self.kv_heads * self.dim, bias=False)
        self.output = nn.Linear(self.heads * self.dim, width, bias=False)
        self.query_norm = (
            Norm(self.dim, spec["epsilon"]) if normalized else nn.Identity()
        )
        self.key_norm = Norm(self.dim, spec["epsilon"]) if normalized else nn.Identity()

    def forward(self, x, positions, visible, context=None, cache=None):
        batch, length = x.shape[:2]
        source = x if context is None else torch.cat((context, x), 1)
        q = self.query_norm(
            self.query(x).view(batch, length, self.heads, self.dim)
        ).transpose(1, 2)
        k = self.key_norm(
            self.key(source).view(batch, -1, self.kv_heads, self.dim)
        ).transpose(1, 2)
        v = self.value(source).view(batch, -1, self.kv_heads, self.dim).transpose(1, 2)
        if context is None:
            key_positions = positions
        else:
            history = torch.arange(context.shape[1], device=x.device)[None].expand(
                batch, -1
            )
            key_positions = torch.cat((history, positions), 1)
        q, k = rotate(q, positions, self.theta), rotate(k, key_positions, self.theta)
        if cache is not None:
            k, v = torch.cat((cache[0], k), 2), torch.cat((cache[1], v), 2)
        saved = (k, v)
        groups = self.heads // self.kv_heads
        y = F.scaled_dot_product_attention(
            q,
            k.repeat_interleave(groups, 1),
            v.repeat_interleave(groups, 1),
            attn_mask=visible,
            dropout_p=0.0,
        )
        return self.output(y.transpose(1, 2).reshape(batch, length, -1)), saved


class Layer(nn.Module):
    def __init__(self, spec, normalized=False, feed_forward=True):
        super().__init__()
        self.norm = Norm(spec["width"], spec["epsilon"])
        self.attention = Attention(spec, normalized)
        self.feed_forward = (
            FeedForward(spec["width"], spec["intermediate"]) if feed_forward else None
        )
        if feed_forward:
            self.post_norm = Norm(spec["width"], spec["epsilon"])

    def forward(self, x, positions, visible, context=None, cache=None):
        y, saved = self.attention(self.norm(x), positions, visible, context, cache)
        x = x + y
        if self.feed_forward is not None:
            x = x + self.feed_forward(self.post_norm(x))
        return x, saved


def specification(config, method, layers, block_size, mask_id):
    required = (
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_hidden_layers",
    )
    if any(not hasattr(config, key) for key in required):
        raise ValueError("Target must be a dense decoder with grouped-query attention")
    scaling = (
        getattr(config, "rope_scaling", None)
        or getattr(config, "rope_parameters", None)
        or {}
    )
    if scaling.get("rope_type", scaling.get("type", "default")) != "default":
        raise ValueError("Scaled rotary embeddings are not supported")
    if getattr(config, "hidden_act", "silu") != "silu":
        raise ValueError("Target must use the silu gated feed-forward architecture")
    if getattr(config, "num_local_experts", 0) or getattr(config, "num_experts", 0):
        raise ValueError("Mixture-of-experts targets are not supported")
    if getattr(config, "use_sliding_window", False) or "sliding_attention" in (
        getattr(config, "layer_types", None) or []
    ):
        raise ValueError("Sliding-window targets are not supported")
    count = config.num_hidden_layers
    width, heads = config.hidden_size, config.num_attention_heads
    kv_heads = getattr(config, "num_key_value_heads", heads)
    dim = getattr(config, "head_dim", None) or width // heads
    if heads % kv_heads or dim % 2 or count < 2:
        raise ValueError("Invalid head dimensions or insufficient target layers")
    taps = sorted({min(count - 1, 1), max(0, count // 2 - 1), max(1, count - 4)})
    return {
        "method": method,
        "width": width,
        "intermediate": config.intermediate_size,
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": dim,
        "epsilon": config.rms_norm_eps,
        "vocabulary": config.vocab_size,
        "theta": 10000.0
        if method == "tree"
        else scaling.get("rope_theta", getattr(config, "rope_theta", 1000000.0)),
        "layers": layers,
        "block_size": block_size,
        "mask_id": mask_id,
        "taps": taps,
        "target_layers": count,
    }


class TreeDraft(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.fusion = FeedForward(spec["width"], spec["intermediate"])
        self.layers = nn.ModuleList(
            [
                Layer(spec, feed_forward=index < spec["layers"] - 1)
                for index in range(spec["layers"])
            ]
        )
        self.post_norm = Norm(spec["width"], spec["epsilon"])
        self.state_branch = FeedForward(spec["width"], spec["intermediate"])
        self.token_branch = FeedForward(spec["width"], spec["intermediate"])
        self.state_norm = Norm(spec["width"], spec["epsilon"])
        self.token_norm = Norm(spec["width"], spec["epsilon"])

    def forward(self, features, embeddings, cache=None):
        batch, length = features.shape[:2]
        past = 0 if cache is None else cache[0][0].shape[2]
        positions = torch.arange(past, past + length, device=features.device)[
            None
        ].expand(batch, -1)
        keys = torch.arange(past + length, device=features.device)
        visible = keys[None, None, None, :] <= positions[:, None, :, None]
        x = features + self.fusion(features, embeddings)
        saved = []
        for index, layer in enumerate(self.layers):
            x, item = layer(
                x, positions, visible, cache=None if cache is None else cache[index]
            )
            saved.append(item)
        normalized = self.post_norm(x)
        return (
            self.state_norm(x + self.state_branch(normalized)),
            self.token_norm(x + self.token_branch(normalized)),
            saved,
        )


class BlockDraft(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.projection = nn.Linear(
            len(spec["taps"]) * spec["width"], spec["width"], bias=False
        )
        self.context_norm = Norm(spec["width"], spec["epsilon"])
        self.layers = nn.ModuleList(
            [Layer(spec, normalized=True) for _ in range(spec["layers"])]
        )
        self.norm = Norm(spec["width"], spec["epsilon"])

    def forward(self, features, embeddings, anchors):
        batch, count = anchors.shape
        size = self.spec["block_size"]
        context = self.context_norm(self.projection(features))
        positions = anchors[..., None] + torch.arange(size, device=features.device)
        length = count * size
        trunk = torch.arange(features.shape[1], device=features.device)
        visible = trunk[None, None, :] < anchors.repeat_interleave(size, 1)[..., None]
        blocks = torch.arange(length, device=features.device) // size
        own = (blocks[:, None] == blocks[None, :]).expand(batch, -1, -1)
        visible = torch.cat((visible, own), -1)[:, None]
        x = embeddings.reshape(batch, length, -1)
        for layer in self.layers:
            x, _ = layer(x, positions.reshape(batch, length), visible, context=context)
        return self.norm(x).reshape(batch, count, size, -1)


def make_draft(spec, device):
    draft = (TreeDraft if spec["method"] == "tree" else BlockDraft)(spec)
    if spec["method"] == "block":
        for module in draft.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
    return draft.to(device)


class LowRankLinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.left = nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device)
        )
        self.right = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device)
        )
        nn.init.kaiming_uniform_(self.left, a=5**0.5)
        self.scale, self.enabled = alpha / rank, True

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            update = F.linear(F.linear(x.to(self.left.dtype), self.left), self.right)
            y = y + (update * self.scale).to(y.dtype)
        return y


def attach_policy_adapter(target, rank, alpha):
    target.requires_grad_(False)
    names = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    selected = [
        (name, module)
        for name, module in target.named_modules()
        if name.rsplit(".", 1)[-1] in names and isinstance(module, nn.Linear)
    ]
    if not selected:
        raise ValueError("Target has no supported attention/feed-forward projections")
    for name, module in selected:
        parent, _, child = name.rpartition(".")
        setattr(target.get_submodule(parent), child, LowRankLinear(module, rank, alpha))


@contextmanager
def reference_policy(target):
    modules = [
        module for module in target.modules() if isinstance(module, LowRankLinear)
    ]
    previous = [module.enabled for module in modules]
    try:
        for module in modules:
            module.enabled = False
        yield
    finally:
        for module, value in zip(modules, previous):
            module.enabled = value


def policy_state(target, adapted):
    if not adapted:
        return target.state_dict()
    return {
        name: param.detach()
        for name, param in target.named_parameters()
        if param.requires_grad
    }


def restore_policy(target, state, adapted):
    if not adapted:
        target.load_state_dict(state)
        return
    expected = {
        name for name, param in target.named_parameters() if param.requires_grad
    }
    if set(state) != expected:
        raise ValueError("Policy adapter checkpoint does not match the target")
    with torch.no_grad():
        for name, param in target.named_parameters():
            if name in state:
                param.copy_(state[name])
