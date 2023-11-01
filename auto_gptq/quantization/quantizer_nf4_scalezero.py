from logging import getLogger

import torch
import torch.nn as nn


logger = getLogger(__name__)


def quantize(x, scale, zero):
    code = torch.tensor([-1, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, 
                        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0, 
                        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 
                        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1], dtype=torch.float16)
    code = code + 1
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

class Quantizer_nf4(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer_nf4, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False, two_scale=False
    ):

        # self.maxq = torch.tensor(2 ** bits - 1)
        self.maxq = torch.tensor(2)
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

        self.scale = (xmax - xmin) / self.maxq
        self.zero = torch.round(-xmin / self.scale)

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
            self.scale2 = torch.zeros_like(self.scale)
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
            return quantize(x, self.scale, self.zero)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


__all__ = ["Quantizer_nf4"]
