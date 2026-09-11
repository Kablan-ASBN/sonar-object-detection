"""Gradient reversal layer and the schedule that controls its strength."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class _GradientReversal(torch.autograd.Function):
    """Identity on the way forward, negated and scaled on the way back."""

    @staticmethod
    def forward(ctx, x: Tensor, coeff: float) -> Tensor:
        ctx.coeff = float(coeff)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        return -ctx.coeff * grad_output, None


def gradient_reverse(x: Tensor, coeff: float) -> Tensor:
    """Return `x` unchanged while multiplying the gradient flowing through it by `-coeff`."""
    return _GradientReversal.apply(x, coeff)


class GradientReversal(nn.Module):
    """Gradient reversal whose coefficient the training loop updates every step."""

    def __init__(self, coeff: float = 0.0) -> None:
        super().__init__()
        self.coeff = float(coeff)

    def set_coeff(self, coeff: float) -> None:
        self.coeff = float(coeff)

    def forward(self, x: Tensor) -> Tensor:
        return gradient_reverse(x, self.coeff)

    def extra_repr(self) -> str:
        return f"coeff={self.coeff:g}"


def ramp(progress: float, max_coeff: float = 1.0, gamma: float = 10.0) -> float:
    """Sigmoid warm-up from 0 to `max_coeff` over `progress` in [0, 1].

    Starting at zero matters: a discriminator that wins early pushes the backbone into
    degenerate features before the detector has learned anything worth aligning.
    """
    # The lower clamp is the one that bites. Negative progress makes the sigmoid term negative,
    # which flips the sign of the reversal and turns the adversary into a collaborator.
    clamped = min(max(progress, 0.0), 1.0)
    return max_coeff * (2.0 / (1.0 + math.exp(-gamma * clamped)) - 1.0)
