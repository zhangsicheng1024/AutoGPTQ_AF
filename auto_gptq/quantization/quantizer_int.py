from logging import getLogger

import torch
import torch.nn as nn


logger = getLogger(__name__)

def quantize(x, scale, zero, code):
    dev = x.device
    shape = x.shape # [in_channel, group_size] in rtn, [in_channel, 1] in gptq
    code = code.to(dev)

    q = x / scale + zero

    q = q.reshape(-1,1) # [in_channel * group_size, 1]
    distance = torch.abs(q - code) # [in_channel * group_size, code_size]
    idx = torch.argmin(distance, dim=-1) # [in_channel * group_size]
    q = torch.gather(code, -1, idx) # [in_channel * group_size]
    q = q.reshape(shape) # [in_channel, group_size]

    xq = (q - zero) * scale

    return xq


class Quantizer_int(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer_int, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False,
        two_scale=False, percentile=1.0, weight=None
    ):
        self.bits = bits
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1)
        self.two_scale = two_scale
        
        self.percentile = percentile
        self.max_value = weight.max()
        self.min_value = weight.min()

        if self.bits == 4:
            self.maxq = torch.tensor(15)
            self.code = torch.tensor([-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.float16)
        elif self.bits == 3:
            self.maxq = torch.tensor(7)
            self.code = torch.tensor([-4, -3, -2, -1, 0, 1, 2, 3], dtype=torch.float16)
        elif self.bits == 2:
            self.maxq = torch.tensor(3)
            self.code = torch.tensor([-2, -1, 0, 1], dtype=torch.float16)

    def find_params(self, x, weight=False, percentile=None):
        if percentile == None: percentile = self.percentile
        x = x.clamp(max=self.max_value*percentile, min=self.min_value*percentile)
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

        if not self.two_scale:
            self.scale = 2 * torch.maximum(torch.abs(xmax), torch.abs(xmin)) / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale) - (2 ** (self.bits-1)) # int4: [0, 15] -> [-8, 7]

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

    def find_params_col(self, x):
        dev = x.device
        shape = x.shape
        self.maxq = self.maxq.to(dev)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        # scale_col = (xmax - xmin) / self.maxq
        # zero_col = torch.round(-xmin / scale_col)
        scale_col = 2 * torch.maximum(torch.abs(xmax), torch.abs(xmin)) / self.maxq
        zero_col = torch.zeros_like(scale_col)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale_col = scale_col.reshape(shape)
        self.zero_col = zero_col.reshape(shape)

    def find_params_row(self, x):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        tmp = torch.zeros(x.shape[1], device=dev)
        xmin = torch.minimum(x.min(0)[0], tmp)
        xmax = torch.maximum(x.max(0)[0], tmp)
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        # self.scale_row = (xmax - xmin) / self.maxq
        # self.zero_row = torch.round(-xmin / self.scale_row)
        self.scale_row = 2 * torch.maximum(torch.abs(xmax), torch.abs(xmin)) / self.maxq
        self.zero_row = torch.zeros_like(self.scale_row)

    def quantize_col(self, x):
        return quantize(x, self.scale_col, self.zero_col, self.code)

    def quantize_row(self, x):
        return quantize(x, self.scale_row, self.zero_row, self.code)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.code)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


__all__ = ["Quantizer_int"]
