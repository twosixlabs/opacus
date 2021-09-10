import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
from torch import autograd
import numpy as np

from opacus import PrivacyEngine


private = True
gp_lambda = 10
batch_size = 5


def calc_gradient_penalties(netD, real_data, fake_data, batch_size, ch, dim, device="cpu"):
    # Modified from https://github.com/jalola/improved-wgan-pytorch/blob/master/training_utils.py
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, ch, dim, dim)
    alpha = alpha.to(device)

    fake_data = fake_data.view(batch_size, ch, dim, dim)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    
    norms = gradients.norm(2, dim=1)
    
    gradient_penalties = ((norms - 1) ** 2)
    return gradient_penalties


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28*28, 50)
        self.l2 = nn.Linear(50, 1)
    
    def forward(self, x):
        return self.l2(self.l1(x.view(-1, 28*28)))
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(10, 50)
        self.l2 = nn.Linear(50, 28*28)
    
    def forward(self, x):
        return self.l2(self.l1(x)).view(-1, 1, 28, 28)
    
D = Discriminator()
G = Generator()

adam = optim.Adam(D.parameters())
    
if private:
    pe = PrivacyEngine(
        D,
        sample_rate=0.01,
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)), # Recommended by Opacus
        noise_multiplier=1e-8,
        max_grad_norm=5.0,
        accum_passes=True
    )
    pe.attach(adam)
    
if not private:
    def mock_forward_hook(module, *argv):
        print("Forward:", module)
    def mock_backward_hook(module, *argv):
        print("Backward:", module)

    for module in D.modules():
        if any(p is not None for p in module.parameters(recurse=False)):
            module.register_forward_hook(mock_forward_hook)
            module.register_backward_hook(mock_backward_hook)
        
def print_grad_samples(model):
    print("grad_samples:")
    for p in model.parameters():
        print(p.grad_sample.shape)
        
for i in range(10):
    adam.zero_grad()
    
    real_in = torch.empty((batch_size, 1, 28, 28)).normal_(0.0, 1.0)
    fake_in = G(torch.empty((batch_size, 10)).normal_(0.0, 1.0))

    fake_loss = D(fake_in).mean()
    real_loss = -D(real_in).mean()
    loss = real_loss + fake_loss
    loss.backward()
    
    print_grad_samples(D)

    if private: pe.module.disable_hooks()
        
    penalties = calc_gradient_penalties(D, real_in, fake_in, batch_size, 1, real_in.size(2), "cpu")
    print("penalties:",penalties)
        
    # Loop through each sample in batch and calculate gradient of penalty with respect to D's parameters for each.
    
    for i in range(batch_size):
        penalty_grad = gp_lambda * autograd.grad(penalties[i], D.parameters(), create_graph=True, retain_graph=True, allow_unused=True)
        with torch.no_grad():
            for i, p in enumerate(D.parameters()):
                # Add penalty to both grad samples (real and fake) so that 
                # Bias layers will have None gradient, so set to zero
                p.grad_sample[0] += torch.unsqueeze(torch.zeros_like(p) if penalty_grad[i] is None else penalty_grad[i], dim=0)
            
    print_grad_samples(D)
    
    if private:
        pe.module.enable_hooks()

    adam.step()
    
    if private: 
        eps, _ = pe.get_privacy_spent(0.0001)
        print("Epsilon:",eps)