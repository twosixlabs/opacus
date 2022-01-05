#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
import os
import types
import warnings
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
from scipy.stats import planck
from torch import Tensor, nn

import numpy as np

from opacus.grad_sample import GradSampleModule
from .grad_sample.utils import accum_grads_across_passes
from opacus.utils.tensor_utils import calc_sample_norms_one_layer

from . import privacy_analysis
from .dp_model_inspector import DPModelInspector
from .layers.dp_ddp import (
    DifferentiallyPrivateDistributedDataParallel,
    average_gradients,
)
from .per_sample_gradient_clip import PerSampleGradientClipper
from .utils import clipping


class TMPrivacyEngine:
    r"""
    The main component of Opacus is the ``PrivacyEngine``.

    To train a model with differential privacy, all you need to do
    is to define a ``PrivacyEngine`` and later attach it to your
    optimizer before running.


    Example:
        This example shows how to define a ``PrivacyEngine`` and to attach
        it to your optimizer.

        >>> import torch
        >>> model = torch.nn.Linear(16, 32)  # An example model
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        >>> privacy_engine = PrivacyEngine(model, sample_rate=0.01, noise_multiplier=1.3, max_grad_norm=1.0)
        >>> privacy_engine.attach(optimizer)  # That's it! Now it's business as usual.
    """

    # flake8: noqa: C901
    def __init__(
        self,
        module: nn.Module,
        *,  # As per PEP 3102, this forces clients to specify kwargs explicitly, not positionally
        batch_size: Optional[int] = None,
        sample_size: Optional[int] = None,
        secure_rng: bool = False,
        batch_first: bool = True,
        target_delta: float = 1e-6,
        target_epsilon: Optional[float] = None,
        epochs: Optional[float] = None,
        rho_per_epoch: float = 1000,
        smooth_sens_t: float = 0.01,
        m_trim: float = 10,
        min_val: float = -1,
        max_val: float = 1,
        sens_compute_bs: int = 256,
        **misc_settings,
    ):
        r"""
        Args:
            module: The Pytorch module to which we are attaching the privacy engine
            alphas: A list of RDP orders
            noise_multiplier: The ratio of the standard deviation of the Gaussian noise to
                the L2-sensitivity of the function to which the noise is added
            max_grad_norm: The maximum norm of the per-sample gradients. Any gradient with norm
                higher than this will be clipped to this value.
            batch_size: Training batch size. Used in the privacy accountant.
            sample_size: The size of the sample (dataset). Used in the privacy accountant.
            sample_rate: Sample rate used to build batches. Used in the privacy accountant.
            secure_rng: If on, it will use ``torchcsprng`` for secure random number generation.
                Comes with a significant performance cost, therefore it's recommended that you
                turn it off when just experimenting.
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor will be ``[batch_size, ..., ...]``.
            target_delta: The target delta. If unset, we will set it for you.
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            **misc_settings: Other arguments to the init
        """

        self.steps = 0

        self.batch_size = batch_size
        self.sample_size = sample_size

        if isinstance(
            module, DifferentiallyPrivateDistributedDataParallel
        ) or isinstance(module, torch.nn.parallel.DistributedDataParallel):
            rank = torch.distributed.get_rank()
            n_replicas = torch.distributed.get_world_size()
        else:
            rank = 0
            n_replicas = 1

        self.module = GradSampleModule(module, accum_passes=True)

        self.target_delta = target_delta
        self.secure_rng = secure_rng
        self.batch_first = batch_first
        self.misc_settings = misc_settings
        self.n_replicas = n_replicas
        self.rank = rank

        self.smooth_sens_t = smooth_sens_t
        self.m_trim = m_trim
        self.min_val = min_val
        self.max_val = max_val
        self.sens_compute_bs = sens_compute_bs

        self.rho_per_epoch = rho_per_epoch
        self.rho_per_weight = rho_per_epoch / sum(p.numel() for p in self.module.parameters() if p.requires_grad)
        self._optimize_sigma()

        self.trimmed_this_batch = False

        self.device = next(module.parameters()).device
        self.steps = 0

        if not self.target_delta:
            if self.sample_size:
                warnings.warn(
                    "target_delta unset. Setting it to an order of magnitude less than 1/sample_size."
                )
                self.target_delta = 0.1 * (1 / self.sample_size)
            else:
                raise ValueError("Please provide a target_delta.")

        if self.secure_rng:
            self.seed = None
            try:
                import torchcsprng as csprng
            except ImportError as e:
                msg = (
                    "To use secure RNG, you must install the torchcsprng package! "
                    "Check out the instructions here: https://github.com/pytorch/csprng#installation"
                )
                raise ImportError(msg) from e

            self.seed = None
            self.random_number_generator = csprng.create_random_device_generator(
                "/dev/urandom"
            )
        else:
            warnings.warn(
                "Secure RNG turned off. This is perfectly fine for experimentation as it allows "
                "for much faster training performance, but remember to turn it on and retrain "
                "one last time before production with ``secure_rng`` turned on."
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.seed = int.from_bytes(os.urandom(8), byteorder="big", signed=True)
                self.random_number_generator = self._set_seed(self.seed)

        self.validator = DPModelInspector()
        self.clipper = None  # lazy initialization in attach

    def state_dict(self):
        return {
            "steps": self.steps,
        }

    def load_state_dict(self, state_dict):
        self.steps = state_dict["steps"]

    def detach(self):
        r"""
        Detaches the privacy engine from optimizer.

        To detach the ``PrivacyEngine`` from optimizer, this method returns
        the model and the optimizer to their original states (i.e. all
        added attributes/methods will be removed).
        """
        # 1. Fix optimizer
        optim = self.optimizer
        optim.step = optim.original_step
        delattr(optim, "privacy_engine")
        delattr(optim, "original_step")
        delattr(optim, "original_zero_grad")
        delattr(optim, "virtual_step")

        # 2. Fix module
        self.module._close()

    def attach(self, optimizer: torch.optim.Optimizer):
        r"""
        Attaches the privacy engine to the optimizer.

        Attaches to the ``PrivacyEngine`` an optimizer object,and injects
        itself into the optimizer's step. To do that it,

        1. Validates that the model does not have unsupported layers.

        2. Adds a pointer to this object (the ``PrivacyEngine``) inside the optimizer.

        3. Moves optimizer's original ``step()`` function to ``original_step()``.

        4. Monkeypatches the optimizer's ``step()`` function to call ``step()`` on
        the query engine automatically whenever it would call ``step()`` for itself.

        Args:
            optimizer: The optimizer to which the privacy engine will attach
        """
        if hasattr(optimizer, "privacy_engine"):
            if optimizer.privacy_engine != self:
                raise ValueError(
                    f"Trying to attach to optimizer: {optimizer}, but that optimizer is "
                    f"already attached to a different Privacy Engine: {optimizer.privacy_engine}."
                )
            else:
                warnings.warn(
                    "Trying to attach twice to the same optimizer. Nothing to do."
                )
                return

        self.validator.validate(self.module)

        def dp_zero_grad(self):
            self.privacy_engine.zero_grad()
            self.original_zero_grad()

        def dp_step(self, closure=None, is_empty=False):
            # When the DDP hooks are activated, there is no need for ``PrivacyEngine.step()``
            # because the clipping and noising are performed by the hooks at the end of the backward pass
            if hasattr(self.privacy_engine.module, "ddp_hooks"):
                # We just update the accountant
                self.privacy_engine.steps += 1

            else:
                self.privacy_engine.step(is_empty)
                if isinstance(
                    self.privacy_engine.module._module,
                    DifferentiallyPrivateDistributedDataParallel,
                ):
                    average_gradients(self.privacy_engine.module)
            self.original_step(closure)

        optimizer.privacy_engine = self

        optimizer.dp_step = types.MethodType(dp_step, optimizer)
        optimizer.original_step = optimizer.step
        optimizer.step = types.MethodType(dp_step, optimizer)

        optimizer.original_zero_grad = optimizer.zero_grad
        optimizer.zero_grad = types.MethodType(dp_zero_grad, optimizer)

        def virtual_step(self):
            if hasattr(self.privacy_engine.module, "ddp_hooks"):
                raise NotImplementedError("DDP hook does not support virtual steps.")
            self.privacy_engine.virtual_step()

        optimizer.virtual_step = types.MethodType(virtual_step, optimizer)

        # create a cross reference for detaching
        self.optimizer = optimizer

    def get_privacy_spent(
        self, target_delta: Optional[float] = None
    ) -> Tuple[float, float]:
        n_epochs = int((self.steps-1) * (self.batch_size // self.sample_size))+1 # round up
        rho = self.rho_per_epoch * n_epochs
        return rho + 2*np.sqrt(rho * np.log(1/target_delta)), target_delta

    def zero_grad(self):
        """
        Resets clippers status.

        Clipper keeps internal gradient per sample in the batch in each
        ``forward`` call of the module, they need to be cleaned before the
        next round.

        If these variables are not cleaned the per sample gradients keep
        being concatenated accross batches. If accumulating gradients
        is intented behavious, e.g. simulating a large batch, prefer
        using ``virtual_step()`` function.
        """
        if self.clipper is not None:
            self.clipper.zero_grad()

    def disable_hooks(self) -> None:
        self.module.forward_hooks_enabled = False
        self.module.backward_hooks_enabled = False

    def enable_hooks(self) -> None:
        self.module.forward_hooks_enabled = True
        self.module.backward_hooks_enabled = True

    def disable_forward_hooks(self):
        self.module.forward_hooks_enabled = False

    def disable_backward_hooks(self):
        self.module.backward_hooks_enabled = False

    def enable_forward_hooks(self):
        self.module.forward_hooks_enabled = True

    def enable_backward_hooks(self):
        self.module.backward_hooks_enabled = True

    def set_accum_passes(self, accum_passes: bool):
        self.module.set_accum_passes(accum_passes)

    def trim_grads(self):
        with torch.no_grad():
            batch_size = next(iter(self.module.parameters())).grad_sample.size(1)

            # Generate indices now so they don't need to be every time (will be xs[idx1] - xs[idx2])
            ks = torch.arange(0, batch_size+1, device=self.device) # distances
            ls = torch.arange(0, batch_size+2, device=self.device)
            # Use all l values then take lower triangular part of matrix (with diagonal shifted by one) to remove values where l > k+1
            idx1 = torch.tril(batch_size - self.m_trim + 1 + ks.reshape(-1, 1) - ls, diagonal=1).clamp(-1, batch_size)
            idx2 = (self.m_trim + 1 - ls).clamp(-1, batch_size).reshape(1, -1)
            scalar = torch.exp(-1 * ks * self.smooth_sens_t)

            for p in (pa for pa in self.module.parameters() if pa.requires_grad):
                # reshape p.grad_sample to shape (num_params_in_layer, batch_size)
                grad_shape = p.grad_sample.shape[2:]
                p.grad_sample = p.grad_sample.reshape(batch_size, -1).transpose(0, 1).clamp(self.min_val, self.max_val).sort(dim=1).values

                num_params = p.grad_sample.size(0)
                p.grad_sample = torch.cat((p.grad_sample, torch.full((num_params, 1), self.max_val, device=self.device), torch.full((num_params, 1), self.min_val, device=self.device)), dim=1)

                # Compute sensitivities
                sensitivities = []
                for batch_of_grads in torch.split(p.grad_sample, self.sens_compute_bs):
                    diffs = torch.tril(batch_of_grads[:,idx1] - batch_of_grads[:,idx2], diagonal=1)
                    inner_max = diffs.max(dim=2).values
                    outer_max = (inner_max*scalar).max(dim=1).values

                    sensitivities.append(outer_max)
                sensitivities = torch.cat(sensitivities, dim=0).reshape(grad_shape)

                # Compute trimmed means from p.grad_sample and put in p.grad
                p.grad = p.grad_sample[:,self.m_trim:batch_size-self.m_trim].mean(dim=1).reshape(grad_shape)
                # Add noise scaled by sensitivities
                p.grad += sensitivities*self.sens_scale * torch.distributions.laplace.Laplace(torch.zeros(grad_shape, device=self.device), torch.ones(grad_shape, device=self.device)).sample() * torch.exp(self.sigma * torch.empty(grad_shape, device=self.device).normal_())
                #print((sensitivities*self.sens_scale * torch.distributions.laplace.Laplace(torch.zeros(grad_shape, device=self.device), torch.ones(grad_shape, device=self.device)).sample() * torch.exp(self.sigma * torch.empty(grad_shape, device=self.device).normal_())).std())

                del sensitivities
                del p.grad_sample

        self.trimmed_this_batch = True

    def step(self, is_empty: bool = False):
        self.steps += 1

        if not self.trimmed_this_batch:
            self.trim_grads()

        self.trimmed_this_batch = False

    def to(self, device: Union[str, torch.device]):
        """
        Moves the privacy engine to the target device.

        Args:
            device : The device on which Pytorch Tensors are allocated.
                See: https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device

        Example:
            This example shows the usage of this method, on how to move the model
            after instantiating the ``PrivacyEngine``.

            >>> model = torch.nn.Linear(16, 32)  # An example model. Default device is CPU
            >>> privacy_engine = PrivacyEngine(model, sample_rate=0.01, noise_multiplier=0.8, max_grad_norm=0.5)
            >>> device = "cuda:3"  # GPU
            >>> model.to(device)  # If we move the model to GPU, we should call the to() method of the privacy engine (next line)
            >>> privacy_engine.to(device)

        Returns:
            The current ``PrivacyEngine``
        """
        self.device = device
        return self

    def virtual_step(self):
        r"""
        Takes a virtual step.

        Virtual batches enable training with arbitrary large batch sizes, while
        keeping the memory consumption constant. This is beneficial, when training
        models with larger batch sizes than standard models.

        Example:
            Imagine you want to train a model with batch size of 2048, but you can only
            fit batch size of 128 in your GPU. Then, you can do the following:

            >>> for i, (X, y) in enumerate(dataloader):
            >>>     logits = model(X)
            >>>     loss = criterion(logits, y)
            >>>     loss.backward()
            >>>     if i % 16 == 15:
            >>>         optimizer.step()    # this will call privacy engine's step()
            >>>         optimizer.zero_grad()
            >>>     else:
            >>>         optimizer.virtual_step()   # this will call privacy engine's virtual_step()

            The rough idea of virtual step is as follows:

            1. Calling ``loss.backward()`` repeatedly stores the per-sample gradients
            for all mini-batches. If we call ``loss.backward()`` ``N`` times on
            mini-batches of size ``B``, then each weight's ``.grad_sample`` field will
            contain ``NxB`` gradients. Then, when calling ``step()``, the privacy engine
            clips all ``NxB`` gradients and computes the average gradient for an effective
            batch of size ``NxB``. A call to ``optimizer.zero_grad()`` erases the
            per-sample gradients.

            2. By calling ``virtual_step()`` after ``loss.backward()``,the ``B``
            per-sample gradients for this mini-batch are clipped and summed up into a
            gradient accumulator. The per-sample gradients can then be discarded. After
            ``N`` iterations (alternating calls to ``loss.backward()`` and
            ``virtual_step()``), a call to ``step()`` will compute the average gradient
            for an effective batch of size ``NxB``.

            The advantage here is that this is memory-efficient: it discards the per-sample
            gradients after every mini-batch. We can thus handle batches of arbitrary size.
        """
        self.clipper.clip()
        self.clipper.accumulate_batch()

    def _local_layer_ddp_hook(
        self, p: torch.Tensor, threshold: float, grad: torch.Tensor
    ):
        """
        Backward hook attached to parameter `p`.
        It replaces `grad` by `new_grad` using the per-sample gradients stored in p.grad_sample

        Args:
            # engine: the privacy engine (to get the DP options and clipping values)
            p: the layer to clip and noise
            threshold: the flat clipping value for that layer
            grad: the gradient (unused, but this argument required to be a valid hook)

        The hook operates like ``PrivacyEngine.step()``, but on a single layer:
            1. clip_and_accumulate
            2. get the clip_values to scale the noise
            3. add the noise
        """

        # Similar to `ConstantPerLayerClipper.pre_step()`
        batch_size = p.grad_sample.shape[0]
        clip_value = self.clipper.norm_clipper.thresholds.norm(2)

        # Similar to `ConstantPerLayerClipper.calc_clipping_factors`)
        norms = calc_sample_norms_one_layer(p.grad_sample)
        per_sample_clip_factor = (threshold / (norms + 1e-6)).clamp(max=1.0)

        # Do the clipping
        summed_grad = self.clipper._weighted_sum(per_sample_clip_factor, p.grad_sample)

        # Accumulate the summed gradient for this mini-batch
        if hasattr(p, "summed_grad"):
            p.summed_grad += summed_grad
        else:
            p.summed_grad = summed_grad

        del p.grad_sample

        # Average (or sum) across the batch
        new_grad = self.clipper._scale_summed_grad(p.summed_grad, batch_size)
        del p.summed_grad

        # Only one GPU adds noise
        if self.rank == 0:
            noise = self._generate_noise(clip_value, new_grad) / batch_size
            new_grad += noise

        return new_grad

    def _set_seed(self, seed: int):
        r"""
        Allows to manually set the seed allowing for a deterministic run. Useful if you want to
        debug.

        WARNING: MANUALLY SETTING THE SEED BREAKS THE GUARANTEE OF SECURE RNG.
        For this reason, this method will raise a ValueError if you had ``secure_rng`` turned on.

        Args:
            seed : The **unsecure** seed
        """
        if self.secure_rng:
            raise ValueError(
                "Seed was manually set on a ``PrivacyEngine`` with ``secure_rng`` turned on."
                "This fundamentally breaks secure_rng, and cannot be allowed. "
                "If you do need reproducibility with a fixed seed, first instantiate the PrivacyEngine "
                "with ``secure_seed`` turned off."
            )
        self.seed = seed

        return (
            torch.random.manual_seed(self.seed)
            if self.device.type == "cpu"
            else torch.cuda.manual_seed(self.seed)
        )

    def _optimize_sigma(self):
        def opt_exp(eps, t, sigma):
            return 5 * (eps / t) * sigma**3 - 5 * sigma**2 - 1

        target_eps = np.sqrt(2*self.rho_per_weight)
        sigma_lower = self.smooth_sens_t / target_eps
        sigma_upper = max(2*self.smooth_sens_t / target_eps, 1/2)

        loss = opt_exp(target_eps, self.smooth_sens_t, np.mean([sigma_lower, sigma_upper]))
        while np.abs(loss) > 0.001:
            if loss < 0:
                sigma_lower = np.mean([sigma_lower, sigma_upper])
            else:
                sigma_upper = np.mean([sigma_lower, sigma_upper])

            loss = opt_exp(target_eps, self.smooth_sens_t, np.mean([sigma_lower, sigma_upper]))

        self.sigma = np.mean([sigma_lower, sigma_upper])
        print(self.sigma)
        self.sens_scale = 1/(np.exp(-(3/2) * self.sigma**2) * (target_eps - (self.smooth_sens_t / self.sigma)))
        print(self.sens_scale)
        self.steps += 5
        print(self.get_privacy_spent(1e-6))
