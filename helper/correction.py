"""Persistent feature projection and cohort-local verification correction."""

import torch


class Correction:
    def __init__(self, head, rank, topk, learning_rate, projector_rate):
        if rank > head.shape[1]:
            raise ValueError("Correction rank exceeds hidden width")
        indices = torch.linspace(0, head.shape[0] - 1, rank, device=head.device).long()
        rows = head.detach().index_select(0, indices).float().T.cpu()
        self.projector = torch.nn.Parameter(
            torch.linalg.qr(rows, mode="reduced").Q.to(head.device)
        )
        self.optimizer = torch.optim.AdamW(
            [self.projector], lr=projector_rate, weight_decay=0.0
        )
        self.topk, self.learning_rate = min(topk, head.shape[0]), learning_rate
        self.matrix = torch.zeros(head.shape[0], rank, device=head.device)
        self.gradient = torch.zeros_like(self.projector)
        self.weight = 0

    @torch.no_grad()
    def reset(self):
        self.matrix.zero_()
        self.gradient.zero_()
        self.weight = 0

    @torch.no_grad()
    def apply(self, hidden, logits):
        return logits.float() + (hidden.float() @ self.projector) @ self.matrix.T

    @torch.no_grad()
    def feedback(self, hidden, logits, probabilities):
        """Use union coordinates; read the old matrix before writing its update."""
        hidden = hidden.reshape(-1, hidden.shape[-1]).float()
        logits = logits.reshape(hidden.shape[0], -1).float()
        probabilities = probabilities.reshape_as(logits).float()
        projected = hidden @ self.projector
        student = (logits + projected @ self.matrix.T).softmax(-1)
        own = student.topk(self.topk, -1).indices
        teacher = probabilities.topk(self.topk, -1).indices
        valid = (probabilities.gather(-1, teacher) > 0) & ~(
            teacher[..., :, None] == own[..., None, :]
        ).any(-1)
        indices = torch.cat((own, teacher), -1)
        valid = torch.cat((torch.ones_like(own, dtype=torch.bool), valid), -1)
        residual = torch.where(
            valid, student.gather(-1, indices) - probabilities.gather(-1, indices), 0.0
        )
        direction = (residual[..., None] * self.matrix[indices]).sum(-2)
        self.gradient.add_(hidden.T @ direction)
        self.weight += hidden.shape[0]
        update = torch.zeros_like(self.matrix)
        update.index_add_(
            0,
            indices.flatten(),
            (residual[..., None] * projected[:, None]).reshape(-1, projected.shape[-1]),
        )
        self.matrix.add_(update, alpha=-self.learning_rate / hidden.shape[0])

    def step(self):
        if self.weight:
            self.optimizer.zero_grad(set_to_none=True)
            self.projector.grad = self.gradient / self.weight
            self.optimizer.step()

    def state_dict(self):
        return {
            "projector": self.projector.detach(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        with torch.no_grad():
            self.projector.copy_(state["projector"])
        self.optimizer.load_state_dict(state["optimizer"])
