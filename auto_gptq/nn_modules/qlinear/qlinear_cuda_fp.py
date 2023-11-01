import math
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import transformers
import torch.functional as F
import time

logger = getLogger(__name__)
try:
    import autogptq_cuda_256
    import autogptq_cuda_64
    _autogptq_cuda_available = True
except ImportError:
    logger.warning('CUDA extension not installed.')
    autogptq_cuda_256 = None
    autogptq_cuda_64 = None
    _autogptq_cuda_available = False

def fptoint(w, scale, code):
    # w: [in_channel]
    # scale: [in_channel]

    q = w / scale
    q = q.reshape(-1,1) # [in_channel, 1]
    distance = torch.abs(q - code) # [in_channel, code_size]
    idx = torch.argmin(distance, dim=-1) # [in_channel]
    q = idx.to(torch.int)
    
    return q

def inttofp(q, scale, code):
    # q: [group_size, in_channel, out_channel]
    # scale: [group_size, in_channel / group_size, out_channel]

    dev = q.device
    shape = q.shape
    code = code.to(dev)

    q = q.reshape(-1).to(torch.int64) # [group_size * in_channel * out_channel]
    w = torch.gather(code, -1, q) # [group_size * in_channel * out_channel]
    w = w.reshape(shape) # [group_size, in_channel, out_channel]
    w = (w * scale).to(torch.float16)

    return w

def fptoint_2scale(w, scale, scale2, code):
    # w: [in_channel]
    # scale: [in_channel]

    w_pos = torch.zeros_like(w)
    w_neg = torch.zeros_like(w)
    w_pos = torch.where(w >= 0, w, w_pos)
    w_neg = torch.where(w < 0, w, w_neg)
    q_pos = w_pos / scale
    q_neg = w_neg / scale2

    q = q_pos + q_neg
    q = q.reshape(-1,1) # [in_channel, 1]
    distance = torch.abs(q - code) # [in_channel, code_size]
    idx = torch.argmin(distance, dim=-1) # [in_channel]
    q = idx.to(torch.int)
    
    return q

def inttofp_2scale(q, scale, scale2, code):
    # q: [group_size, in_channel, out_channel]
    # scale: [group_size, in_channel / group_size, out_channel]

    dev = q.device
    shape = q.shape
    code = code.to(dev)

    q = q.reshape(-1).to(torch.int64) # [group_size * in_channel * out_channel]
    w = torch.gather(code, -1, q) # [group_size * in_channel * out_channel]
    w = w.reshape(shape) # [group_size, in_channel, out_channel]
    
    w_pos = torch.zeros_like(w)
    w_neg = torch.zeros_like(w)
    w_pos = torch.where(w >= 0, w, w_pos)
    w_neg = torch.where(w < 0, w, w_neg)
    w = (w_pos * scale + w_neg * scale2).to(torch.float16)

    return w

class QuantLinear(nn.Module):
    def __init__(
        self,
        bits,
        group_size,
        infeatures,
        outfeatures,
        bias,
        two_scale=False,
        use_cuda_fp16=True,
        kernel_switch_threshold=128,
        trainable=False,
        weight_dtype=torch.float16,
    ):
        super().__init__()
        global _autogptq_cuda_available
        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        if trainable:
            _autogptq_cuda_available = False
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2 ** self.bits - 1

        self.register_buffer(
            'qweight',
            torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32)
        )
        self.register_buffer(
            'scales',
            torch.zeros((math.ceil(infeatures / self.group_size), outfeatures), dtype=torch.float16)
        )
        self.register_buffer(
            'scales2',
            torch.zeros((math.ceil(infeatures / self.group_size), outfeatures), dtype=torch.float16)
        )
        self.register_buffer(
            'g_idx',
            torch.tensor([i // self.group_size for i in range(infeatures)], dtype=torch.int32)
        )

        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None
        self.half_indim = self.infeatures // 2

        self.use_cuda_fp16 = use_cuda_fp16 if bits != 8 else False
        
        # is performed by unpacking the weights and using torch.matmul
        if self.bits in [2, 4, 8]:
            self.wf = torch.tensor(list(range(0, 32, self.bits)), dtype=torch.int32).unsqueeze(0)
        elif self.bits == 3:
            self.wf = torch.tensor(
                [
                    [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                    [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                    [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],
                ],
                dtype=torch.int32
            ).reshape(1, 3, 12)

        self.kernel_switch_threshold = kernel_switch_threshold
        self.autogptq_cuda_available = _autogptq_cuda_available
        self.autogptq_cuda = autogptq_cuda_256
        if infeatures % 256 != 0 or outfeatures % 256 != 0:
            self.autogptq_cuda = autogptq_cuda_64
        if infeatures % 64 != 0 or outfeatures % 64 != 0:
            self.autogptq_cuda_available = False

        self.trainable = trainable

        self.two_scale = two_scale
        if self.bits == 4:
            self.code = torch.tensor([-6, -4, -3, -2, -1.5, -1, -0.5, -0, 0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float16)
        elif self.bits == 3:
            self.code = torch.tensor([-4, -2, -1, -0, 0, 1, 2, 4], dtype=torch.float16)

    def pack(self, linear, scales, scales2, g_idx):
        W = linear.weight.data.clone()
        if isinstance(linear, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.t()

        scales = scales.t().contiguous()
        scales2 = scales2.t().contiguous()
        self.scales = scales.clone().half()
        self.scales2 = scales2.clone().half()
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        intweight = []
        code = self.code.to(W.device)
        for idx in range(self.infeatures):
            g_idx = idx // self.group_size
            if self.two_scale:
                q = fptoint_2scale(W[:, idx], self.scales[g_idx], self.scales2[g_idx], code)[:, None]
            else:
                q = fptoint(W[:, idx], self.scales[g_idx], code)[:, None]
            intweight.append(q)
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        i = 0
        row = 0
        qweight = np.zeros(
            (intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32
        )
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape(-1, x.shape[-1])
        if self.autogptq_cuda_available is True and (
            self.kernel_switch_threshold is False or x.shape[0] < self.kernel_switch_threshold
        ):
            raise NotImplementedError('cuda kernel unimplemented')
        else:
            if self.wf.device != self.scales.device:
               self.wf = self.wf.to(self.scales.device)
                
            if self.bits in [2,4,8]:
   
               scales = self.scales
               scales = scales.reshape(-1, 1, scales.shape[-1])
               scales2 = self.scales2
               scales2 = scales2.reshape(-1, 1, scales2.shape[-1])
                
               weight = torch.bitwise_right_shift(torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1), self.wf.unsqueeze(-1)).to(torch.int16 if self.bits == 8 else torch.int8)
               torch.bitwise_and(weight,(2 ** self.bits) - 1, out=weight)
               weight = weight.reshape(-1, self.group_size, weight.shape[2])
            elif self.bits == 3:
                # raise NotImplementedError("3 bits unimplemented")
                
                scales = self.scales
                scales = scales.reshape(-1, 1, scales.shape[-1])
                scales2 = self.scales2
                scales2 = scales2.reshape(-1, 1, scales2.shape[-1])
                
                weight = self.qweight.reshape(self.qweight.shape[0]//3, 3, 1, self.qweight.shape[1]).expand(-1, -1, 12, -1)
                weight = (weight >> self.wf.unsqueeze(-1))&0x7
                weight[:,0,10] = (weight[:,0,10]&0x3) | ((weight[:,1,0] << 2)&0x4)
                weight[:,1,11] = (weight[:,1,11]&0x1) | ((weight[:,2,0] << 1)&0x6)
                weight = weight & 0x7
                weight = torch.cat([weight[:,0,:11], weight[:,1,1:12], weight[:,2,1:11]], dim=1)
                weight = weight.reshape(-1, self.group_size, weight.shape[2])
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")
            if self.two_scale:
                weight = inttofp_2scale(weight, scales, scales2, self.code)
            else:
                weight = inttofp(weight, scales, self.code)
            weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

            out = torch.matmul(x.half(), weight)
        out = out.half().reshape(out_shape)
        out = out + self.bias if self.bias is not None else out
        return out


__all__ = ["QuantLinear"]
