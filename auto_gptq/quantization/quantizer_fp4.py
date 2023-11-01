from logging import getLogger

import torch
import torch.nn as nn


logger = getLogger(__name__)

def quantize(x, scale, code):
    dev = x.device
    shape = x.shape # [in_channel, group_size] in rtn, [in_channel, 1] in gptq
    code = code.to(dev)

    q = x / scale

    q = q.reshape(-1,1) # [in_channel * group_size, 1]
    distance = torch.abs(q - code) # [in_channel * group_size, code_size]
    idx = torch.argmin(distance, dim=-1) # [in_channel * group_size]
    q = torch.gather(code, -1, idx) # [in_channel * group_size]
    q = q.reshape(shape) # [in_channel, group_size]

    xq = q * scale

    return xq

def quantize_2scale(x, scale_pos, scale_neg, code):
    dev = x.device
    shape = x.shape # [in_channel, group_size] in rtn, [in_channel, 1] in gptq
    code = code.to(dev)

    x_pos = torch.zeros_like(x)
    x_neg = torch.zeros_like(x)
    x_pos = torch.where(x >= 0, x, x_pos)
    x_neg = torch.where(x < 0, x, x_neg)
    q_pos = x_pos / scale_pos
    q_neg = x_neg / scale_neg

    q_pos = q_pos.reshape(-1,1) # [in_channel * group_size, 1]
    distance = torch.abs(q_pos - code) # [in_channel * group_size, code_size]
    idx = torch.argmin(distance, dim=-1) # [in_channel * group_size]
    q_pos = torch.gather(code, -1, idx) # [in_channel * group_size]
    q_pos = q_pos.reshape(shape) # [in_channel, group_size]

    q_neg = q_neg.reshape(-1,1)
    distance = torch.abs(q_neg - code)
    idx = torch.argmin(distance, dim=-1)
    q_neg = torch.gather(code, -1, idx)
    q_neg = q_neg.reshape(shape)

    xq = q_pos * scale_pos + q_neg * scale_neg
    return xq


class Quantizer_fp4(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer_fp4, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('scale2', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False,
        two_scale=False
    ):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.two_scale = two_scale
        if trits:
            self.maxq = torch.tensor(-1)

        if self.bits == 4:
            self.maxq = torch.tensor(12)
            self.code = torch.tensor([-6, -4, -3, -2, -1.5, -1, -0.5, -0, 0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float16)
        elif self.bits == 3:
            self.maxq = torch.tensor(8)
            self.code = torch.tensor([-4, -2, -1, -0, 0, 1, 2, 4], dtype=torch.float16)

    def find_params(self, x, weight=False):
        dev = x.device
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

        if not self.two_scale:
            self.scale = torch.maximum(torch.abs(xmax), torch.abs(xmin)) / (self.maxq / 2)
            self.scale2 = torch.zeros_like(self.scale)
        else:
            self.scale = torch.abs(xmax) / (self.maxq / 2)
            self.scale2 = torch.abs(xmin) / (self.maxq / 2)

        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.scale2 = self.scale2.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.scale2 = self.scale2.unsqueeze(0)


    def quantize(self, x):
        if self.ready():
            if self.two_scale:
                return quantize_2scale(x, self.scale, self.scale2, self.code)
            else:
                return quantize(x, self.scale, self.code)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)

__all__ = ["Quantizer_fp4"]
