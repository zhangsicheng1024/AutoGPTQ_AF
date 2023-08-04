from logging import getLogger

import torch
import torch.nn as nn
from .ffi import *


logger = getLogger(__name__)

# Seems this simple RTN fp quant doesn't perform well, use ppq fp quant CUDA kernel instead
# def quantize(x, scale, zero, maxq):
#     dev = x.device
#     q = x / scale
#     q = torch.where(q >= 5,                             torch.tensor(6.).to(dev), q)
#     q = torch.where((q < 5)         & (q >= 3.5),       torch.tensor(4.).to(dev), q)
#     q = torch.where((q < 3.5)       & (q >= 2.5),       torch.tensor(3.).to(dev), q)
#     q = torch.where((q < 2.5)       & (q >= 1.75),      torch.tensor(2.).to(dev), q)
#     q = torch.where((q < 1.75)      & (q >= 1.25),      torch.tensor(1.5).to(dev), q)
#     q = torch.where((q < 1.25)      & (q >= 0.75),     torch.tensor(1.).to(dev), q)
#     q = torch.where((q < 0.75)     & (q >= 0.25),     torch.tensor(0.5).to(dev), q)
#     q = torch.where((q < 0.25)     & (q >= -0.25),    torch.tensor(0.).to(dev), q)
#     q = torch.where((q < -0.25)    & (q >= -0.75),    torch.tensor(-0.5).to(dev), q)
#     q = torch.where((q < -0.75)    & (q >= -1.25),     torch.tensor(-1.).to(dev), q)
#     q = torch.where((q < -1.25)     & (q >= -1.75),     torch.tensor(-1.5).to(dev), q)
#     q = torch.where((q < -1.75)     & (q >= -2.5),      torch.tensor(-2.).to(dev), q)
#     q = torch.where((q < -2.5)      & (q >= -3.5),      torch.tensor(-3.).to(dev), q)
#     q = torch.where((q < -2.5)      & (q >= -5),        torch.tensor(-4.).to(dev), q)
#     q = torch.where(q < -5,                             torch.tensor(-6.).to(dev), q)
#     return scale * q

class Quantizer_fp4(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer_fp4, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False, exponet_bits=2, mantissa_bits=1
    ):
        self.maxq = torch.tensor(2*2**(2**exponet_bits/2) * (2-(0.5)**(mantissa_bits))) # 12
        self.quant_min = - 2**(2**exponet_bits/2) * (2-(0.5)**(mantissa_bits)) # -6
        self.quant_max = 2**(2**exponet_bits/2) * (2-(0.5)**(mantissa_bits)) # 6
        self.exponet_bits = exponet_bits
        self.mantissa_bits = mantissa_bits

        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1)

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        # =========zyj============
        self.scale = (xmax - xmin) / self.maxq
        # self.zero = torch.round(- ( xmin + self.maxq / 2 ) / self.scale)
        self.zero = torch.zeros_like(self.scale)
        # ========================

        if self.mse:
            raise NotImplementedError("fp4 mse not implemented")
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            # return quantize(x, self.scale, self.zero, self.maxq)
            # x, scale: [768, 1]
            x = CUDA.FloatingQuantize_T(
                tensor=x,
                scales=self.scale,
                offsets=self.zero,
                exponent=self.exponet_bits,
                mantissa=self.mantissa_bits,
                minimum=self.quant_min,
                maximum=self.quant_max,
                rounding=0
            )

            # is equal to quant + dequant:

            # q = CUDA.FloatingQuantize_T_Quant(
            #     tensor=x,
            #     scales=self.scale,
            #     offsets=self.zero,
            #     exponent=self.exponet_bits,
            #     mantissa=self.mantissa_bits,
            #     minimum=self.quant_min,
            #     maximum=self.quant_max,
            #     rounding=0
            # )
            # x = CUDA.FloatingQuantize_T_Dequant(
            #     tensor=q,
            #     scales=self.scale,
            #     offsets=self.zero,
            #     exponent=self.exponet_bits,
            #     mantissa=self.mantissa_bits,
            #     minimum=self.quant_min,
            #     maximum=self.quant_max,
            #     rounding=0
            # )
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


__all__ = ["Quantizer_fp4"]
