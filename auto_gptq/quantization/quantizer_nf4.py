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

# def quantize_3bit(x, scale): 
#     dev = x.device
#     q = x / scale
#     q = torch.where(q >= 0.8114928305149078, torch.tensor(1.0).to(dev), q) 
#     q = torch.where((q < 0.8114928305149078) & (q >= 0.5024898052215576), torch.tensor(0.6229856610298157).to(dev), q) 
#     q = torch.where((q < 0.5024898052215576) & (q >= 0.2826657369732857), torch.tensor(0.3819939494132996).to(dev), q) 
#     q = torch.where((q < 0.2826657369732857) & (q >= 0.0916687622666359), torch.tensor(0.1833375245332718).to(dev), q) 
#     q = torch.where((q < 0.0916687622666359) & (q >= -0.1234657019376755), torch.tensor(0).to(dev), q) 
#     q = torch.where((q < -0.1234657019376755) & (q >= -0.39097706973552704), torch.tensor(-0.2469314038753510).to(dev), q) 
#     q = torch.where((q < -0.39097706973552704) & (q >= -0.7675113677978516), torch.tensor(-0.5350227355957031).to(dev), q) 
#     q = torch.where(q < -0.7675113677978516, torch.tensor(-1.0).to(dev), q) 
#     return q * scale

# def quantize_3bit_2scale(x, scale_pos, scale_neg):
#     dev = x.device
#     x_pos = torch.zeros_like(x)
#     x_neg = torch.zeros_like(x)
#     x_pos = torch.where(x >= 0, x, x_pos)
#     x_neg = torch.where(x < 0, x, x_neg)
#     q_pos = x_pos / scale_pos
#     q_neg = x_neg / scale_neg

#     q_pos = torch.where(q_pos >= 0.8114928305149078,                                    torch.tensor(1.0).to(dev), q_pos) 
#     q_pos = torch.where((q_pos < 0.8114928305149078) & (q_pos >= 0.5024898052215576),   torch.tensor(0.6229856610298157).to(dev), q_pos) 
#     q_pos = torch.where((q_pos < 0.5024898052215576) & (q_pos >= 0.2826657369732857),   torch.tensor(0.3819939494132996).to(dev), q_pos) 
#     q_pos = torch.where((q_pos < 0.2826657369732857) & (q_pos >= 0.0916687622666359),   torch.tensor(0.1833375245332718).to(dev), q_pos) 
#     q_pos = torch.where(q_pos < 0.0916687622666359,                                     torch.tensor(0).to(dev), q_pos) 

#     q_neg = torch.where(q_neg >= -0.1234657019376755,                                       torch.tensor(0).to(dev), q_neg) 
#     q_neg = torch.where((q_neg < -0.1234657019376755) & (q_neg >= -0.39097706973552704),    torch.tensor(-0.2469314038753510).to(dev), q_neg) 
#     q_neg = torch.where((q_neg < -0.39097706973552704) & (q_neg >= -0.7675113677978516),    torch.tensor(-0.5350227355957031).to(dev), q_neg) 
#     q_neg = torch.where(q_neg < -0.7675113677978516,                                        torch.tensor(-1.0).to(dev), q_neg) 

#     q = q_pos * scale_pos + q_neg * scale_neg
#     return q

# def quantize_4bit(x, scale):
#     dev = x.device
#     q = x / scale
#     q = torch.where(q >= 0.8614784181118011,                                    torch.tensor(1.0).to(dev), q)
#     q = torch.where((q < 0.8614784181118011)    & (q >= 0.6427869200706482),    torch.tensor(0.7229568362236023).to(dev), q)
#     q = torch.where((q < 0.6427869200706482)    & (q >= 0.5016634166240692),    torch.tensor(0.5626170039176941).to(dev), q)
#     q = torch.where((q < 0.5016634166240692)    & (q >= 0.3893125355243683),    torch.tensor(0.44070982933044434).to(dev), q)
#     q = torch.where((q < 0.3893125355243683)    & (q >= 0.2920137718319893),    torch.tensor(0.33791524171829224).to(dev), q)
#     q = torch.where((q < 0.2920137718319893)    & (q >= 0.2035212516784668),    torch.tensor(0.24611230194568634).to(dev), q)
#     q = torch.where((q < 0.2035212516784668)    & (q >= 0.1202552504837513),    torch.tensor(0.16093020141124725).to(dev), q)
#     q = torch.where((q < 0.1202552504837513)    & (q >= 0.03979014977812767),   torch.tensor(0.07958029955625534).to(dev), q)
#     q = torch.where((q < 0.03979014977812767)   & (q >= -0.045525018125772476), torch.tensor(0).to(dev), q)
#     q = torch.where((q < -0.045525018125772476) & (q >= -0.13791173323988914),  torch.tensor(-0.09105003625154495).to(dev), q)
#     q = torch.where((q < -0.13791173323988914)  & (q >= -0.23460740596055984),  torch.tensor(-0.18477343022823334).to(dev), q)
#     q = torch.where((q < -0.23460740596055984)  & (q >= -0.33967943489551544),  torch.tensor(-0.28444138169288635).to(dev), q)
#     q = torch.where((q < -0.33967943489551544)  & (q >= -0.4599952697753906),   torch.tensor(-0.39491748809814453).to(dev), q)
#     q = torch.where((q < -0.4599952697753906)   & (q >= -0.6106329262256622),   torch.tensor(-0.5250730514526367).to(dev), q)
#     q = torch.where((q < -0.6106329262256622)   & (q >= -0.8480964004993439),   torch.tensor(-0.6961928009986877).to(dev), q)
#     q = torch.where(q < -0.8480964004993439,                                    torch.tensor(-1.0).to(dev), q)
#     return q * scale 

# def quantize_4bit_2scale(x, scale_pos, scale_neg):
#     dev = x.device
#     x_pos = torch.zeros_like(x)
#     x_neg = torch.zeros_like(x)
#     x_pos = torch.where(x >= 0, x, x_pos)
#     x_neg = torch.where(x < 0, x, x_neg)
#     q_pos = x_pos / scale_pos
#     q_neg = x_neg / scale_neg

#     q_pos = torch.where(q_pos >= 0.8614784181118011,                                        torch.tensor(1.0).to(dev), q_pos)
#     q_pos = torch.where((q_pos < 0.8614784181118011)    & (q_pos >= 0.6427869200706482),    torch.tensor(0.7229568362236023).to(dev), q_pos)
#     q_pos = torch.where((q_pos < 0.6427869200706482)    & (q_pos >= 0.5016634166240692),    torch.tensor(0.5626170039176941).to(dev), q_pos)
#     q_pos = torch.where((q_pos < 0.5016634166240692)    & (q_pos >= 0.3893125355243683),    torch.tensor(0.44070982933044434).to(dev), q_pos)
#     q_pos = torch.where((q_pos < 0.3893125355243683)    & (q_pos >= 0.2920137718319893),    torch.tensor(0.33791524171829224).to(dev), q_pos)
#     q_pos = torch.where((q_pos < 0.2920137718319893)    & (q_pos >= 0.2035212516784668),    torch.tensor(0.24611230194568634).to(dev), q_pos)
#     q_pos = torch.where((q_pos < 0.2035212516784668)    & (q_pos >= 0.1202552504837513),    torch.tensor(0.16093020141124725).to(dev), q_pos)
#     q_pos = torch.where((q_pos < 0.1202552504837513)    & (q_pos >= 0.03979014977812767),   torch.tensor(0.07958029955625534).to(dev), q_pos)
#     q_pos = torch.where(q_pos < 0.03979014977812767,                                        torch.tensor(0).to(dev), q_pos)

#     q_neg = torch.where(q_neg >= -0.045525018125772476,                                     torch.tensor(0).to(dev), q_neg)
#     q_neg = torch.where((q_neg < -0.045525018125772476) & (q_neg >= -0.13791173323988914),  torch.tensor(-0.09105003625154495).to(dev), q_neg)
#     q_neg = torch.where((q_neg < -0.13791173323988914)  & (q_neg >= -0.23460740596055984),  torch.tensor(-0.18477343022823334).to(dev), q_neg)
#     q_neg = torch.where((q_neg < -0.23460740596055984)  & (q_neg >= -0.33967943489551544),  torch.tensor(-0.28444138169288635).to(dev), q_neg)
#     q_neg = torch.where((q_neg < -0.33967943489551544)  & (q_neg >= -0.4599952697753906),   torch.tensor(-0.39491748809814453).to(dev), q_neg)
#     q_neg = torch.where((q_neg < -0.4599952697753906)   & (q_neg >= -0.6106329262256622),   torch.tensor(-0.5250730514526367).to(dev), q_neg)
#     q_neg = torch.where((q_neg < -0.6106329262256622)   & (q_neg >= -0.8480964004993439),   torch.tensor(-0.6961928009986877).to(dev), q_neg)
#     q_neg = torch.where(q_neg < -0.8480964004993439,                                        torch.tensor(-1.0).to(dev), q_neg)

#     q = q_pos * scale_pos + q_neg * scale_neg
#     return q

class Quantizer_nf4(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer_nf4, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))

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
            self.code = torch.tensor([-1, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, 
                                      -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0, 
                                      0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 
                                      0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1], dtype=torch.float16)
        elif self.bits == 3:
            self.code = torch.tensor([-1, -0.5350227355957031, -0.2469314038753510, 0, 
                                      0.1833375245332718, 0.3819939494132996, 0.6229856610298157, 1], dtype=torch.float16)

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


        # =========zyj============
        self.scale = torch.maximum(torch.abs(xmax), torch.abs(xmin))
        self.scale_pos = torch.abs(xmax)
        self.scale_neg = torch.abs(xmin)


        # ========================
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.scale_pos = self.scale_pos.reshape(shape)
            self.scale_neg = self.scale_neg.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.scale_pos = self.scale_pos.unsqueeze(0)
            self.scale_neg = self.scale_neg.unsqueeze(0)


    def quantize(self, x):
        if self.ready():
            if self.two_scale:
                return quantize_2scale(x, self.scale_pos, self.scale_neg, self.code)
            else:
                return quantize(x, self.scale, self.code)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)

__all__ = ["Quantizer_nf4"]
