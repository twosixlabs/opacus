import torch
from torch import nn
import numpy as np
import types
import warnings
import os
from functools import partial
from typing import List, Optional, Tuple, Union

from opacus import privacy_analysis
from opacus import PrivacyEngine
from opacus.layers.dp_ddp import (
    DifferentiallyPrivateDistributedDataParallel,
    average_gradients,
)


DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

class ISPrivacyEngine(PrivacyEngine):
    def __init__(
        self,
        module: nn.Module,
        *,  # As per PEP 3102, this forces clients to specify kwargs explicitly, not positionally
        sample_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        sample_size: Optional[int] = None,
        noise_multiplier: Optional[float] = None,
        alphas: List[float] = DEFAULT_ALPHAS,
        secure_rng: bool = False,
        batch_first: bool = True,
        target_delta: float = 1e-6,
        target_epsilon: Optional[float] = None,
        epochs: Optional[float] = None,
        poisson: bool = False,
        per_param: bool = False,
        **misc_settings,
    ):

        self.steps = 0
        self.poisson = poisson
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self._set_sample_rate()

        self.per_param = per_param
        self.batch_sensitivity = []

        if isinstance(
            module, DifferentiallyPrivateDistributedDataParallel
        ) or isinstance(module, torch.nn.parallel.DistributedDataParallel):
            rank = torch.distributed.get_rank()
            n_replicas = torch.distributed.get_world_size()
            self.sample_rate *= n_replicas
        else:
            rank = 0
            n_replicas = 1

        self.module = module

        if poisson:
            # TODO: Check directly if sampler is UniformSampler when sampler gets passed to the Engine (in the future)
            if sample_size is None:
                raise ValueError(
                    "If using Poisson sampling, sample_size should get passed to the PrivacyEngine."
                )

            # Number of empty batches follows a geometric distribution
            # Planck is the same distribution but its parameter is the (negative) log of the geometric's parameter
            self._poisson_empty_batches_distribution = planck(
                -math.log(1 - self.sample_rate) * self.sample_size
            )

        if noise_multiplier is None:
            if target_epsilon is None or target_delta is None or epochs is None:
                raise ValueError(
                    "If noise_multiplier is not specified, (target_epsilon, target_delta, epochs) should be given to the engine."
                )
            self.noise_multiplier = get_noise_multiplier(
                target_epsilon, target_delta, self.sample_rate, epochs, alphas
            )
        else:
            self.noise_multiplier = noise_multiplier

        self.alphas = alphas
        self.target_delta = target_delta
        self.secure_rng = secure_rng
        self.batch_first = batch_first
        self.misc_settings = misc_settings
        self.n_replicas = n_replicas
        self.rank = rank

        self.device = next(module.parameters()).device
        self.steps = 0

        if self.noise_multiplier < 0:
            raise ValueError(
                f"noise_multiplier={self.noise_multiplier} is not a valid value. Please provide a float >= 0."
            )

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
            self.original_step(closure)

        def poisson_dp_step(self, closure=None):
            # Perform one step as usual
            self.dp_step(closure)

            # Taking empty steps to simulate empty batches
            num_empty_batches = self.privacy_engine._sample_poisson_empty_batches()
            for _ in range(num_empty_batches):
                self.zero_grad()
                self.dp_step(closure, is_empty=True)

        optimizer.privacy_engine = self

        optimizer.dp_step = types.MethodType(dp_step, optimizer)
        optimizer.original_step = optimizer.step

        optimizer.step = types.MethodType(
            poisson_dp_step if self.poisson else dp_step, optimizer
        )

        optimizer.original_zero_grad = optimizer.zero_grad
        optimizer.zero_grad = types.MethodType(dp_zero_grad, optimizer)

        def virtual_step(self):
            if hasattr(self.privacy_engine.module, "ddp_hooks"):
                raise NotImplementedError("DDP hook does not support virtual steps.")
            self.privacy_engine.virtual_step()

        optimizer.virtual_step = types.MethodType(virtual_step, optimizer)

        # create a cross reference for detaching
        self.optimizer = optimizer

        if self.poisson:
            # Optional initial step on empty batch
            num_empty_batches = self._sample_poisson_empty_batches()
            for _ in range(num_empty_batches):
                self.optimizer.zero_grad()
                for p in self.module.parameters():
                    if p.requires_grad:
                        p.grad = torch.zeros_like(p)
                self.optimizer.dp_step(closure=None, is_empty=True)

    def _sample_poisson_empty_batches(self):
        """
        Samples an integer which is equal to the number of (consecutive) empty batches when doing Poisson sampling
        """
        return self._poisson_empty_batches_distribution.rvs(size=1)[0]

    def get_renyi_divergence(self):
        rdp = torch.tensor(
            privacy_analysis.compute_rdp(
                self.sample_rate, self.noise_multiplier, 1, self.alphas
            )
        )
        return rdp

    def get_privacy_spent(
        self, target_delta: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Computes the (epsilon, delta) privacy budget spent so far.

        This method converts from an (alpha, epsilon)-DP guarantee for all alphas that
        the ``PrivacyEngine`` was initialized with. It returns the optimal alpha together
        with the best epsilon.

        Args:
            target_delta: The Target delta. If None, it will default to the privacy
                engine's target delta.

        Returns:
            Pair of epsilon and optimal order alpha.
        """
        if target_delta is None:
            if self.target_delta is None:
                raise ValueError(
                    "If self.target_delta is not specified, target_delta should be set as argument to get_privacy_spent."
                )
            target_delta = self.target_delta
        rdp = self.get_renyi_divergence() * self.steps
        eps, best_alpha = privacy_analysis.get_privacy_spent(
            self.alphas, rdp, target_delta
        )
        return float(eps), float(best_alpha)

    def zero_grad(self):
        self.batch_sensitivity = 0

    def backward(self, loss, inputs):
        # (1) first-order gradient (wrt parameters)
        first_order_grads = torch.autograd.grad(loss, self.module.parameters(), retain_graph=True, create_graph=True)
        # with torch.no_grad():
        #     for i, p in enumerate(self.module.parameters()):
        #         p.grad = first_order_grads[i]

        if self.per_param:
            self.batch_sensitivity = []
            grad_l2_norms = torch.stack([x.view(-1).norm(2) for x in first_order_grads], dim=0)

            for param_grad_l2_norm in grad_l2_norms:
                sensitivity_vec = torch.autograd.grad(param_grad_l2_norm, inputs=inputs, retain_graph=True)
                s = [torch.norm(v, p=2).cpu().numpy().item() for v in sensitivity_vec]

                self.batch_sensitivity.append(np.max(s))

            print(self.batch_sensitivity)
        else:
            # (2) L2 norm of the gradient from (1)
            grad_l2_norm = torch.norm(torch.cat([x.view(-1) for x in first_order_grads]), p=2)

            # (3) Gradient (wrt inputs) of the L2 norm of the gradient from (2)
            sensitivity_vec = torch.autograd.grad(grad_l2_norm, inputs, retain_graph=True)[0]

            # (4) L2 norm of (3) - "immediate sensitivity"
            s = [torch.norm(v, p=2).cpu().numpy().item() for v in sensitivity_vec]

            self.batch_sensitivity = np.max(s)

    def step(self, is_empty: bool = False):
        self.steps += 1

        params = (p for p in self.module.parameters() if p.requires_grad)
        for i, p in enumerate(params):
            sens = self.batch_sensitivity[i] if self.per_param else self.batch_sensitivity
            noise = self._generate_noise(sens * self.noise_multiplier, p)

            if self.rank == 0:
                # Noise only gets added on first worker
                # This is easy to reason about for loss_reduction=sum
                # For loss_reduction=mean, noise will get further divided by
                # world_size as gradients are averaged.
                p.grad += noise

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
            noise = self._generate_noise(clip_value, new_grad)
            if self.loss_reduction == "mean":
                noise /= batch_size
            new_grad += noise

        # Poisson uses avg_batch_size instead of batch_size
        if self.poisson and self.loss_reduction == "mean":
            new_grad *= batch_size / self.avg_batch_size

        return new_grad

    def _register_ddp_hooks(self):
        """
        Adds hooks for DP training over DistributedDataParallel.

        Each layer has a hook that clips and noises the gradients as soon as they are ready.
        """

        # `thresholds` is a tensor with `len(params)` thresholds (i.e. max layer norm)
        params = (p for p in self.module.parameters() if p.requires_grad)
        thresholds = self.clipper.norm_clipper.thresholds

        # Register and store the DDP hooks (one per layer). GradSampleModule knows how to remove them.
        self.module.ddp_hooks = []
        for p, threshold in zip(params, thresholds):
            if not p.requires_grad:
                continue

            self.module.ddp_hooks.append(
                p.register_hook(partial(self._local_layer_ddp_hook, p, threshold))
            )

    def _generate_noise(
        engine, max_grad_norm: float, grad: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Generates a tensor of Gaussian noise of the same shape as ``grad``.

        The generated tensor has zero mean and standard deviation
        sigma = ``noise_multiplier x max_grad_norm ``

        Args:
            max_grad_norm : The maximum norm of the per-sample gradients.
            grad : The gradient of the reference, based on which the dimension of the
                noise tensor will be determined

        Returns:
            the generated noise with noise zero and standard
            deviation of ``noise_multiplier x max_grad_norm ``
        """
        if engine.noise_multiplier > 0 and max_grad_norm > 0:
            return torch.normal(
                0,
                engine.noise_multiplier * max_grad_norm,
                grad.shape,
                device=engine.device,
                generator=engine.random_number_generator,
            )
        return torch.zeros(grad.shape, device=engine.device)

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

    def _set_sample_rate(self):
        r"""
        Determine the ``sample_rate``.

        If a ``sample_rate`` is provided, it will be used.
        If no ``sample_rate``is provided, the used ``sample_rate`` will be equal to
        ``batch_size`` /  ``sample_size``.
        """
        if self.batch_size and not isinstance(self.batch_size, int):
            raise ValueError(
                f"batch_size={self.batch_size} is not a valid value. Please provide a positive integer."
            )

        if self.sample_size and not isinstance(self.sample_size, int):
            raise ValueError(
                f"sample_size={self.sample_size} is not a valid value. Please provide a positive integer."
            )

        if self.sample_rate is None:
            if self.batch_size is None or self.sample_size is None:
                raise ValueError(
                    "You must provide (batch_size and sample_sizes) or sample_rate."
                )
            else:
                self.sample_rate = self.batch_size / self.sample_size
                if self.batch_size is not None or self.sample_size is not None:
                    warnings.warn(
                        "The sample rate will be defined from ``batch_size`` and ``sample_size``."
                        "The returned privacy budget will be incorrect."
                    )

            self.avg_batch_size = self.sample_rate * self.sample_size
        else:
            warnings.warn(
                "A ``sample_rate`` has been provided."
                "Thus, the provided ``batch_size``and ``sample_size`` will be ignored."
            )
            if self.poisson:
                if self.loss_reduction == "mean" and not self.sample_size:
                    raise ValueError(
                        "Sample size has to be provided if using Poisson and loss_reduction=mean."
                    )
                self.avg_batch_size = self.sample_rate * self.sample_size

        if self.sample_rate > 1.0:
            raise ValueError(
                f"sample_rate={self.sample_rate} is not a valid value. Please provide a float between 0 and 1."
            )
