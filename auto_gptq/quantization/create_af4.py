import torch
import sys

# ==============af4 int===========================
# def create_afint_numbers(xmax, xmin):
#     result = []
#     for i in range(len(xmax)):
#         pos = xmax[i]
#         neg = xmin[i]
#         if abs(pos) > abs(neg):
#             pos_points = torch.linspace(pos, 0, 9)
#             neg_points = torch.linspace(neg, 0, 8)[:-1]
#         else:
#             pos_points = torch.linspace(pos, 0, 8)
#             neg_points = torch.linspace(neg, 0, 9)[:-1]
#         result.append(pos_points)
#         result.append(neg_points)
#     result = torch.cat(result)
#     return result

def create_afint_numbers(xmax, xmin, percentile_max=0.9):
    # TODO: matmul accelerate
    result = []
    for i in range(len(xmax)):
        pos = xmax[i] * percentile_max
        neg = xmin[i] * percentile_max
        if abs(pos) > abs(neg):
            pos_points = torch.linspace(pos, 0, 9)
            neg_points = torch.linspace(neg, 0, 8)[:-1]
        else:
            pos_points = torch.linspace(pos, 0, 8)
            neg_points = torch.linspace(neg, 0, 9)[:-1]
        result.append(pos_points)
        result.append(neg_points)
    result = torch.cat(result)
    result = torch.reshape(result, (len(xmax), 16))
    return result

# ==============af4 fp===========================
# def create_affp_numbers(xmax, xmin):
#     result = []
#     for i in range(len(xmax)):
#         pos_scale = xmax[i] / 6
#         neg_scale = xmin[i] / 6
#         code_affp = torch.tensor([6*pos_scale, 4*pos_scale, 3*pos_scale, 2*pos_scale, 1.5*pos_scale, 1*pos_scale, 0.5*pos_scale, 0, 0, 0.5*neg_scale, 1*neg_scale, 1.5*neg_scale, 2*neg_scale, 3*neg_scale, 4*neg_scale, 6*neg_scale])
#         result.append(code_affp)
#     result = torch.cat(result)
#     return result

def create_affp_numbers(xmax, xmin, percentile_max=0.9):
    dtype = xmax.dtype
    dev = xmax.device
    pos_scale = xmax * percentile_max / 6
    neg_scale = xmin * percentile_max / 6
    pos_base = torch.tensor([6, 4, 3, 2, 1.5, 1, 0.5, 0]).to(dtype).to(dev)
    neg_base = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6]).to(dtype).to(dev)
    pos_code = torch.matmul(pos_scale.unsqueeze(1), pos_base.unsqueeze(0))
    neg_code = torch.matmul(neg_scale.unsqueeze(1), neg_base.unsqueeze(0))
    return torch.cat((pos_code, neg_code), 1)
    # result = []
    # for i in range(len(xmax)):
    #     pos_scale = xmax[i] * percentile_max / 6
    #     neg_scale = xmin[i] * percentile_max / 6
    #     code_affp = torch.tensor([6*pos_scale, 4*pos_scale, 3*pos_scale, 2*pos_scale, 1.5*pos_scale, 1*pos_scale, 0.5*pos_scale, 0, 0, 0.5*neg_scale, 1*neg_scale, 1.5*neg_scale, 2*neg_scale, 3*neg_scale, 4*neg_scale, 6*neg_scale])
    #     result.append(code_affp)
    # result = torch.cat(result)
    # return result