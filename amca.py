
import math

import torch
from torch.optim import Optimizer


class AMCA(Optimizer):
    """
    Adaptive Momentum with Cosine Annealing (AMCA).

    beta_t = beta0 * min(1, gamma / (||g_t|| + eps))
    eta_t  = eta_min + 0.5 * (eta_max - eta_min)
             * (1 + cos(pi * t / total_steps))
    """

    def __init__(
        self,
        params,
        lr_max=1e-3,
        lr_min=1e-5,
        beta0=0.9,
        gamma=1.0,
        eps=1e-8,
        total_steps=10000,
    ):
        if total_steps < 1:
            raise ValueError("total_steps must be at least 1")
        if lr_max <= 0 or lr_min < 0 or lr_min > lr_max:
            raise ValueError("Require lr_max > 0 and 0 <= lr_min <= lr_max")

        defaults = dict(
            lr=lr_max,
            lr_max=lr_max,
            lr_min=lr_min,
            beta0=beta0,
            gamma=gamma,
            eps=eps,
        )
        super().__init__(params, defaults)
        self.total_steps = total_steps
        self.step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Calculate one global gradient norm across all parameters.
        grad_sq_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grad = p.grad.detach().float()
                    grad_sq_sum += torch.sum(grad * grad).item()

        grad_norm = math.sqrt(grad_sq_sum)

        group0 = self.param_groups[0]
        beta0 = group0["beta0"]
        gamma = group0["gamma"]
        eps = group0["eps"]

        beta_t = beta0 * min(
            1.0, gamma / (grad_norm + eps)
        )

        t = min(self.step_count, self.total_steps)

        eta_t = (
            group0["lr_min"]
            + 0.5
            * (group0["lr_max"] - group0["lr_min"])
            * (1 + math.cos(math.pi * t / self.total_steps))
        )

        for group in self.param_groups:
            group["lr"] = eta_t

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)

                momentum = state["momentum_buffer"]
                momentum.mul_(beta_t).add_(
                    grad, alpha=1.0 - beta_t
                )

                p.add_(momentum, alpha=-eta_t)

        self.step_count += 1
        return loss
