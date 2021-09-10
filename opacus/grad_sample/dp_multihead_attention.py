#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import torch
from typing import Callable
from opacus.layers.dp_multihead_attention import SequenceBias

from .utils import create_or_extend_grad_sample
from .register_grad_sampler import register_grad_sampler


@register_grad_sampler(SequenceBias)
def compute_sequence_bias_grad_sample(
    layer: SequenceBias, A: torch.Tensor, B: torch.Tensor,
    add_grad_sample_fn: Callable[[torch.tensor, torch.tensor, int], None],
    batch_dim: int = 0,
) -> None:
    """
    Computes per sample gradients for ``SequenceBias`` layer

    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
        batch_dim: Batch dimension position
    """
    add_grad_sample_fn(layer.bias, B[:, -1], batch_dim)
