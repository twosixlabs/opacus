#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Callable
import torch
import torch.nn as nn

from .utils import create_or_extend_grad_sample
from .register_grad_sampler import register_grad_sampler


@register_grad_sampler(nn.Linear)
def compute_linear_grad_sample(
    layer: nn.Linear, A: torch.Tensor, B: torch.Tensor,
    add_grad_sample_fn: Callable[[torch.tensor, torch.tensor, int], None],
    batch_dim: int = 0,
) -> None:
    """
    Computes per sample gradients for ``nn.Linear`` layer

    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
        batch_dim: Batch dimension position
    """
    gs = torch.einsum("n...i,n...j->nij", B, A)
    add_grad_sample_fn(layer.weight, gs, batch_dim)
    if layer.bias is not None:

        add_grad_sample_fn(
            layer.bias,
            torch.einsum("n...k->nk", B),
            batch_dim,
        )
