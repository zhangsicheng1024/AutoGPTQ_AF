from logging import getLogger

import torch
import torch.nn as nn
from .ffi import *
from .create_af4 import *


logger = getLogger(__name__)

def quantize(x, scale_pos, scale_neg, code):
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

def quantize_2bit(x, code):
    # x: [in_channel, group_size]
    # code: [in_channel, 4]
    group_size = x.shape[1]
    x_ = x.unsqueeze(-1).repeat(1,1,4) # [in_channel, group_size, 4]
    code_ = code.unsqueeze(1).repeat(1, group_size, 1) # [in_channel, group_size, 4]
    distance = torch.abs(x_ - code_) # [in_channel, group_size, 4]
    idx = torch.argmin(distance, dim=-1) # [in_channel, group_size]
    xq = torch.gather(code, 1, idx) # [in_channel, group_size]

    return xq

class Quantizer_af4(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer_af4, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        # TODO 2 scale

    def configure(
        self,
        bits, perchannel=False, sym=True, 
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False,
        tensor_percentile=1.0, group_percentile=1.0, format_prototype='fp',
        weight=None
    ):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink 
        self.tensor_percentile = tensor_percentile
        self.group_percentile = group_percentile
        self.format_prototype = format_prototype

        max_value = weight.max()
        min_value = weight.min()
        self.tensor_max = max_value * self.tensor_percentile
        self.tensor_min = min_value * self.tensor_percentile

        # if trits:
        #     self.maxq = torch.tensor(-1) 

        if self.bits == 4:
            self.maxq = torch.tensor(12)
            self.code = torch.tensor([-6, -4, -3, -2, -1.5, -1, -0.5, -0, 0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float16)
        elif self.bits == 3:
            self.maxq = torch.tensor(8)
            self.code = torch.tensor([-4, -2, -1, -0, 0, 1, 2, 4], dtype=torch.float16)

    def find_params(self, x, weight=False):
        x = x.clamp(max=self.tensor_max, min=self.tensor_min)
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

        self.xmax = xmax
        self.xmin = xmin

        self.group_max, _ = torch.max(x, dim=1)
        self.group_min, _ = torch.min(x, dim=1)

        # TODO: calculate scale and return for pack
        self.scale = torch.maximum(torch.abs(xmax), torch.abs(xmin))

        # for 4bit/3bit 2 scale, [in_channel, 1]
        self.scale_pos = torch.abs(xmax) / (self.maxq / 2)
        self.scale_neg = torch.abs(xmin) / (self.maxq / 2)
        
        # for 2bit, self.code_2bit: [in_channel, 4]
        abs_max = torch.where(xmax.abs()>xmin.abs(), xmax, xmin)
        abs_max = abs_max / 2
        self.code_2bit = torch.cat((xmax.unsqueeze(1), xmin.unsqueeze(1), abs_max.unsqueeze(1), torch.zeros_like(xmax).unsqueeze(1)), 1)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.scale_pos = self.scale_pos.reshape(shape)
            self.scale_neg = self.scale_neg.reshape(shape)
            return

    def quantize(self, x):
        x = x.clamp(max=self.tensor_max, min=self.tensor_min)
        if(self.bits == 2):
            return quantize_2bit(x, self.code_2bit)
        return quantize(x, self.scale_pos, self.scale_neg, self.code)

        # x = x.clamp(max=self.tensor_max, min=self.tensor_min)
        # if self.format_prototype == "int":
        #     code = create_afint_numbers(self.group_max, self.group_min, self.group_percentile).cuda()
        # elif self.format_prototype == "fp":
        #     code = create_affp_numbers(self.bits, self.group_max, self.group_min, self.group_percentile).cuda()
        # distances = torch.abs(x.unsqueeze(-1) - code.unsqueeze(1))
        # idx = torch.argmin(distances, dim=-1)
        # q = torch.gather(code, -1, idx)
        # return q

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


__all__ = ["Quantizer_af4"]
