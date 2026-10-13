"""Low-rank policy adapters and policy checkpoint restoration."""

from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


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
