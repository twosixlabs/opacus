#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import create_or_extend_grad_sample
from .register_grad_sampler import register_grad_sampler


@register_grad_sampler(
    [
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    ]
)
def compute_instance_norm_grad_sample(
    layer: Union[
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    ],
    A: torch.Tensor,
    B: torch.Tensor,
    add_grad_sample_fn: Callable[[torch.tensor, torch.tensor, int], None],
    batch_dim: int = 0,
) -> None:
    """
    Computes per sample gradients for InstanceNorm layers

    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
        batch_dim: Batch dimension position
    """
    gs = F.instance_norm(A, eps=layer.eps) * B
    add_grad_sample_fn(layer.weight, torch.einsum("ni...->ni", gs), batch_dim)
    if layer.bias is not None:
        add_grad_sample_fn(
            layer.bias, torch.einsum("ni...->ni", B), batch_dim
        )
