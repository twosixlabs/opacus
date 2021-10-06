<p align="center"><img src="https://github.com/pytorch/opacus/blob/main/website/static/img/opacus_logo.svg" alt="Opacus" width="500"/></p>

<hr/>

# Calvin's README
Changes from original Opacus:
1. Immediate sensitivity implementation as an alternate privacy engine (ISPrivacyEngine)

ISPrivacyEngine shares the same accounting and many other features as the normal PrivacyEngine, but requires that you call .backward() through the privacy engine.
This is to avoid computing the gradients twice when calculating IS.

2. Slight rework of how per-sample gradients are handled and stored

In the original Opacus implementation, a new "sample" is added to parameter.grad_sample each time a backward hook on that parameter is triggered.
This causes some undesireable behavior when multiple backward calls happen in an iteration, such as when training a GAN (fake loss and real loss).
In the [Opacus DCGAN example](https://github.com/calvinhirsch/opacus/blob/calvin-branch/examples/dcgan.py), they address this by doing two optimizer steps per iteration.
However, this may lead to differing optimizer performance from a standard training formulation.
In addition, the privacy engine accounting then counts steps only on fake (generated) data, unnecessarily doubling the step count in the accounting.
This implementation changes parameter.grad_sample from shape (n_batch, *) to shape (n_passes, n_batch, *), where n_passes is the number of backward passes through that parameter.
In addition, it offers two modes: accum_passes={True, False}.
Either the gradients can be accumulated (summed) across passes and then the summed per-sample gradients can be clipped or the per-sample gradients from different passes can be stored separately, clipped separately, and then summed (this method requires passing a value for the number of these passes that contain sensitive data in order to scale the noise accordingly).

These methods can be found implemented in my [pytorch implementation of DP-CGAN](), where basic clipping shows accum_passes=True and split clipping shows accum_passes=False.

3. Made it easier to customize private training procedure

Default Opacus behavior is to clip the per-sample gradients and accumulate them into summed_grad before anything else when .step() is called.
This implementation moves clip() and accumulate_batch() to their own functions that will either be called automatically, preserving default behavior, when auto_clip_and_accum_on_step is left as True, or setting it to false allows them to be called manually instead. In addition, when specifying accum_passes=False, .accum_passes() can be called manually.

4. Added moving average clipping to privacy engine

Added a simple moving average for gradient norm multiplied by some constant at each layer that adaptively changes clipping parameter over time. Alternate adaptive clipping strategies can be implemented by calling privacy_engine.set_max_grad_norm().

<hr/>

# Orignal README

[![CircleCI](https://circleci.com/gh/pytorch/opacus.svg?style=svg)](https://circleci.com/gh/pytorch/opacus)

[Opacus](https://opacus.ai) is a library that enables training PyTorch models with differential privacy. It supports training with minimal code changes required on the client, has little impact on training performance and allows the client to online track the privacy budget expended at any given moment.

## Target audience
This code release is aimed at two target audiences:
1. ML practitioners will find this to be a gentle introduction to training a model with differential privacy as it requires minimal code changes.
2. Differential Privacy scientists will find this easy to experiment and tinker with, allowing them to focus on what matters.


## Installation
The latest release of Opacus can be installed via `pip`:
```bash
pip install opacus
```

> :warning: **NOTE**: This will bring in the latest version of our deps, which are on Cuda 10.2. This will not work if you environment is using an older Cuda version (for example, Google Colab is still on Cuda 10.1).

To install on Colab, run this cell first:

```bash
pip install torchcsprng==0.1.3+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
Then you can just `pip install opacus` like before. See more context in [this issue](https://github.com/pytorch/opacus/issues/69).


You can also install directly from the source for the latest features (along with its quirks and potentially ocassional bugs):
```bash
git clone https://github.com/pytorch/opacus.git
cd opacus
pip install -e .
```

## Getting started
To train your model with differential privacy, all you need to do is to declare a `PrivacyEngine` and attach it to your optimizer before running, eg:

```python
model = Net()
optimizer = SGD(model.parameters(), lr=0.05)
privacy_engine = PrivacyEngine(
    model,
    sample_rate=0.01,
    alphas=[10, 100],
    noise_multiplier=1.3,
    max_grad_norm=1.0,
)
privacy_engine.attach(optimizer)
# Now it's business as usual
```

The [MNIST example](https://github.com/pytorch/opacus/tree/main/examples/mnist.py) shows an end-to-end run using opacus. The [examples](https://github.com/pytorch/opacus/tree/main/examples/) folder contains more such examples.

## FAQ
Checkout the [FAQ](https://opacus.ai/docs/faq) page for answers to some of the most frequently asked questions about Differential Privacy and Opacus.

## Contributing
See the [CONTRIBUTING](https://github.com/pytorch/opacus/tree/main/CONTRIBUTING.md) file for how to help out.

Do also check out our README files inside the repo to learn how the code is organized.

## References
* [Mironov, Ilya. "Rényi differential privacy." 2017 IEEE 30th Computer Security Foundations Symposium (CSF). IEEE, 2017.](https://arxiv.org/abs/1702.07476)
* [Abadi, Martin, et al. "Deep learning with differential privacy." Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security. ACM, 2016.](https://arxiv.org/abs/1607.00133)
* [Mironov, Ilya, Kunal Talwar, and Li Zhang. "R\'enyi Differential Privacy of the Sampled Gaussian Mechanism." arXiv preprint arXiv:1908.10530 (2019).](https://arxiv.org/abs/1908.10530)
* [Goodfellow, Ian. "Efficient per-example gradient computations." arXiv preprint arXiv:1510.01799 (2015).](https://arxiv.org/abs/1510.01799)
* [McMahan, H. Brendan, and Galen Andrew. "A general approach to adding differential privacy to iterative training procedures." arXiv preprint arXiv:1812.06210 (2018).](https://arxiv.org/abs/1812.06210)

## License
This code is released under Apache 2.0, as found in the [LICENSE](https://github.com/pytorch/opacus/tree/main/LICENSE) file.
