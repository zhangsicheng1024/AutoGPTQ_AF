from logging import getLogger

import torch
import torch.nn as nn
from .ffi import *
from .create_af4 import *


logger = getLogger(__name__)

class Quantizer_af4(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer_af4, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))

    def configure(
        self,
        bits, perchannel=False, sym=True, 
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False
    ):
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
        # print(dev)
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
    
    # def quantize(self, x, group_size, percentile, format_prototype):
    #     if format_prototype == "int":
    #         code = create_afint_numbers(self.xmax, self.xmin, percentile).cuda()
    #     elif format_prototype == "fp":
    #         code = create_affp_numbers(self.xmax, self.xmin, percentile).cuda()
    #     code = torch.reshape(code, (x.shape[0], 16))
    #     distances = torch.abs(x.unsqueeze(-1) - code.unsqueeze(1))
    #     idx = torch.argmin(distances, dim=-1)
    #     x = torch.gather(code, -1, idx)
    #     return x

    def quantize(self, x, group_size, percentile, format_prototype):
        if group_size == -1:
            if format_prototype == "int":
                code = create_afint_numbers(self.xmax, self.xmin, percentile).cuda()
            elif format_prototype == "fp":
                code = create_affp_numbers(self.xmax, self.xmin, percentile).cuda()
            code = torch.reshape(code, (x.shape[0], 16))
            distances = torch.abs(x.unsqueeze(-1) - code.unsqueeze(1))
            idx = torch.argmin(distances, dim=-1)
            x = torch.gather(code, -1, idx)
            return x
        else:
            split_tensors = torch.split(x, group_size, dim=1)
            for i, split_tensor in enumerate(split_tensors):
                # 找到最大值和最小值
                max_val, _ = torch.max(split_tensor, dim=1)
                min_val, _ = torch.min(split_tensor, dim=1)

                if format_prototype == "int":
                    code = create_afint_numbers(max_val, min_val, percentile).cuda()
                elif format_prototype == "fp":
                    code = create_affp_numbers(max_val, min_val, percentile).cuda()
                code = torch.reshape(code, (split_tensor.shape[0], 16))
                distances = torch.abs(split_tensor.unsqueeze(-1) - code.unsqueeze(1))
                idx = torch.argmin(distances, dim=-1)
                x_split_q = torch.gather(code, -1, idx)
                x[:, i*group_size:(i+1)*group_size] = x_split_q
            return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


__all__ = ["Quantizer_af4"]
