<p align="center"><img src="https://github.com/pytorch/opacus/blob/main/website/static/img/opacus_logo.svg" alt="Opacus" width="500"/></p>

<hr/>

# CSL GAN Opacus Fork


## Changes from original Opacus

1. Slight rework of how per-sample gradients are handled and stored

In the original Opacus implementation, a new "sample" is added to parameter.grad_sample each time a backward hook on that parameter is triggered.
This causes some undesireable behavior when multiple backward calls happen in an iteration, such as when training a GAN (fake loss and real loss).
In the [Opacus DCGAN example](https://github.com/calvinhirsch/opacus/blob/calvin-branch/examples/dcgan.py), they address this by doing two optimizer steps per iteration.
However, this may lead to differing optimizer performance from a standard training formulation.
In addition, the privacy engine accounting then counts steps only on fake (generated) data, unnecessarily doubling the step count in the accounting.
This implementation changes parameter.grad_sample from shape (n_batch, *) to shape (n_passes, n_batch, *), where n_passes is the number of backward passes through that parameter.
In addition, it offers two modes: accum_passes={True, False}.
Either the gradients can be accumulated (summed) across passes and then the summed per-sample gradients can be clipped or the per-sample gradients from different passes can be stored separately, clipped separately, and then summed (this method requires passing a value for the number of these passes that contain sensitive data in order to scale the noise accordingly).

These methods can be found implemented in my [pytorch implementation of DP-CGAN](https://github.com/calvinhirsch/dp-cgan-pytorch), where basic clipping shows accum_passes=True and split clipping shows accum_passes=False.

2. Made it easier to customize private training procedure

Default Opacus behavior is to clip the per-sample gradients and accumulate them into summed_grad before anything else when .step() is called.
This implementation moves clip() and accumulate_batch() to their own functions that will either be called automatically, preserving default behavior, when auto_clip_and_accum_on_step is left as True, or setting it to false allows them to be called manually instead. In addition, when specifying accum_passes=False, .accum_passes() can be called manually.


## Additions

The immediate sensitivity privacy engine only approximates differential privacy. The other privacy engines are experimental.
