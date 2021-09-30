#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn


def create_or_extend_grad_sample(
    param: torch.Tensor, grad_sample: torch.Tensor, batch_dim: int
) -> None:
    """
    Creates a ``grad_sample`` attribute in the given parameter, or appends to it
    if the ``grad_sample`` attribute already exists.

    Args:
        param: Parameter to which ``grad_sample`` will be added
        grad_sample: Per-sample gradients tensor. Must be of the same
            shape as ``param`` with extra batch dimension
        batch_dim: Position of the batch dimension in the shape of
            ``grad_sample``
    """

    if hasattr(param, "grad_sample"):
        param.grad_sample = torch.cat((param.grad_sample, torch.unsqueeze(grad_sample, dim=0)), batch_dim)
    else:
        param.grad_sample = torch.unsqueeze(grad_sample, dim=0)


def create_or_accumulate_grad_sample(
    param: torch.Tensor, grad_sample: torch.Tensor, batch_dim: int #layer: nn.Module
) -> None:
    """
    Creates a ``grad_sample`` attribute in the given parameter, or adds to it
    if the ``grad_sample`` attribute already exists.

    Args:
        param: Parameter to which ``grad_sample`` will be added
        grad_sample: Per-sample gradients tensor. Must be of the same
            shape as ``param`` with extra batch dimension
    """

    if hasattr(param, "grad_sample") and param.grad_sample.size(0) > 0:
        if param.grad_sample.size(0) > 1:
            raise Exception("create_or_accumulate_grad_sample called after create_or_extend_grad_sample without calling accum_grads_across_passes between.")

        param.grad_sample[0] += grad_sample
    else:
        # TO-DO: What is the point of this?
#         param.grad_sample = torch.zeros(
#             torch.Size([max_batch_len]) + grad_sample.shape[1:],
#             grad_sample.shape,
#             device=grad_sample.device,
#             dtype=grad_sample.dtype,
#         )
        param.grad_sample = torch.unsqueeze(grad_sample, dim=0).detach()

def accum_grads_across_passes(param: torch.Tensor) -> None:
    if param.grad_sample.size(0) > 1:
        for i in range(1, param.grad_sample.size(0)):
            param.grad_sample[0] += param.grad_sample[i]
        param.grad_sample = torch.unsqueeze(param.grad_sample[0], dim=0)
    else:
        raise Exception("One or fewer ({}) previous passes of grad_samples are stored, nothing to accumulate!".format(param.grad_sample.size(0)))
