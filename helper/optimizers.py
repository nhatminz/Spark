"""FP32 optimizer weights and gradient accumulation for mixed precision."""

import torch


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
