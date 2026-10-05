import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from torch import nn
import julius

from torch.amp import autocast
from modules.activation_functions import *


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


@torch.jit.script
def snakebeta(x, alpha, beta):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (beta + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class SnakeBeta(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        return snakebeta(x, torch.exp(self.alpha), torch.exp(self.beta))


class ResampleUpsampler(nn.Module):
    def __init__(self, n_mel, upsample_rate, c_prev, c_cur, upps):
        super(ResampleUpsampler, self).__init__()
        self.scale_factor = upsample_rate
        self.upps = upps
        self.convolution_after = weight_norm(nn.Conv1d(c_prev, c_cur, 1, 1))
        self.convolution_noise = weight_norm(nn.Conv1d(n_mel, c_prev, 7, 1, padding=3))

    def forward(self, x, x0):
        B, C, T = x0.shape
        y0 = torch.zeros(B, C, T * self.upps, device=x0.device)
        y0[:, :, :: self.upps] = x0
        y0 = self.convolution_noise(y0)
        with autocast("cuda", enabled=False):
            y0 = julius.highpass_filter(y0.float(), 0.5 / self.scale_factor)

        B, C, T = x.shape
        y = torch.zeros(B, C, T * self.scale_factor, device=x.device)
        y[:, :, :: self.scale_factor] = x
        with autocast("cuda", enabled=False):
            y = julius.lowpass_filter(y.float(), 0.5 / self.scale_factor)

        y0 = y0.to(x0.dtype)
        y = y.to(x.dtype)
        y = y + y0
        y = self.convolution_after(y)
        return y
