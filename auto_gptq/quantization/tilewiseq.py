import torch
import sys

def calculate_snr(tensor1, tensor2):
    # 将输入张量转换为PyTorch张量
    tensor1 = torch.tensor(tensor1, dtype=torch.float32)
    tensor2 = torch.tensor(tensor2, dtype=torch.float32)

    # 计算信号的平方和
    signal_power = torch.sum(tensor1**2)

    # 计算噪声的平方和
    noise_tensor = tensor1 - tensor2
    noise_power = torch.sum(noise_tensor**2)

    # 计算SNR
    snr = 10 * torch.log10(signal_power / noise_power)

    return snr.item()  # 将结果转换为标量并返回

def torch_mean_square_error(y_pred: torch.Tensor, y_real: torch.Tensor, reduction: str='mean') -> torch.Tensor:
    if y_pred.shape != y_real.shape:
        raise ValueError(f'Can not compute mse loss for tensors with different shape. '
            f'({y_pred.shape} and {y_real.shape})')
    reduction = str(reduction).lower()

    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(0)
        y_real = y_real.unsqueeze(0)

    diff = torch.pow(y_pred.float() - y_real.float(), 2).flatten(start_dim=1)
    mse  = torch.mean(diff, dim=-1)

    if reduction == 'mean':
        return torch.mean(mse)
    elif reduction == 'sum':
        return torch.sum(mse)
    elif reduction == 'none':
        return mse
    else:
        raise ValueError(f'Unsupported reduction method.')

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

    return xq, x / q

def scale_initial_sicheng(W_tile):
    W_flatten = W_tile.clone().flatten(1)
    tmp = torch.zeros(W_flatten.shape[1], device = W_tile.device)
    xmin = torch.minimum(W_flatten.min(0)[0], tmp)
    xmax = torch.maximum(W_flatten.max(0)[0], tmp)
    tmp = (xmin == 0) & (xmax == 0)
    xmin[tmp] = -1
    xmax[tmp] = +1
    scale_col = torch.maximum(torch.abs(xmax), torch.abs(xmin))
    

    W_col_q = W_flatten / scale_col
    scale_col = torch.unsqueeze(scale_col, 0)

    tmp = torch.zeros(W_col_q.shape[0], device = W_tile.device)
    xmin = torch.minimum(W_col_q.min(1)[0], tmp)
    xmax = torch.maximum(W_col_q.max(1)[0], tmp)
    tmp = (xmin == 0) & (xmax == 0)
    xmin[tmp] = -1
    xmax[tmp] = +1
    scale_row = torch.maximum(torch.abs(xmax), torch.abs(xmin))

    scale_row = torch.unsqueeze(scale_row, 1)

    max_singular_value = 1
    return scale_row, scale_col, max_singular_value

def scale_initial_sicheng_t(W_tile):  # 转置后的tensor处理
    W_flatten = W_tile.clone().flatten(1)
    tmp = torch.zeros(W_flatten.shape[0], device = W_tile.device)
    xmin = torch.minimum(W_flatten.min(1)[0], tmp)
    xmax = torch.maximum(W_flatten.max(1)[0], tmp)
    tmp = (xmin == 0) & (xmax == 0)
    xmin[tmp] = -1
    xmax[tmp] = +1
    scale_row = torch.maximum(torch.abs(xmax), torch.abs(xmin))


    scale_row = torch.unsqueeze(scale_row, 1)
    W_row_q = W_flatten / scale_row

    tmp = torch.zeros(W_row_q.shape[1], device = W_tile.device)
    xmin = torch.minimum(W_row_q.min(0)[0], tmp)
    xmax = torch.maximum(W_row_q.max(0)[0], tmp)
    tmp = (xmin == 0) & (xmax == 0)
    xmin[tmp] = -1
    xmax[tmp] = +1
    scale_col = torch.maximum(torch.abs(xmax), torch.abs(xmin))

    scale_col = torch.unsqueeze(scale_col, 0)

    max_singular_value = 1
    return scale_row, scale_col, max_singular_value

def tilewiseq(W, tile_size_row, tile_size_col):
    num_blocks_row = W.shape[0] // tile_size_row
    num_blocks_col = W.shape[1] // tile_size_col
    code = torch.tensor([-1, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, 
                            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0, 
                            0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 
                            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1], dtype=torch.float16)
    for i in range(num_blocks_row):
        for j in range(num_blocks_col):
            start_row = i * tile_size_row
            end_row = (i + 1) * tile_size_row
            start_col = j * tile_size_col
            end_col = (j + 1) * tile_size_col


            W_tile = W[start_row:end_row, start_col:end_col].clone()

            scale_row, scale_col, max_singular_value = scale_initial_sicheng_t(W_tile)

            min_loss = 100
            for k in range(100):
                scale = max_singular_value * scale_row.view(-1, 1) @ scale_col.view(1, -1)
                W_q, scale_new = quantize(W_tile, scale, code)
                loss = torch_mean_square_error(W_q, W_tile)
                if loss < min_loss:
                    min_loss = loss
                    scale_new_min = scale
                scale_new = scale_new.to(torch.float)
                inf_indices = torch.isinf(scale_new)
                scale_new[inf_indices] = scale[inf_indices] 
                nan_indices = torch.isnan(scale_new)
                scale_new[nan_indices] = scale[nan_indices] 

                U, S, Vt = torch.svd(scale_new)
                max_singular_value = S[0]
                scale_row = U[:, 0]
                scale_col = Vt.t()[0]
            U, S, Vt = torch.svd(scale_new_min)
            max_singular_value = S[0]
            scale_row = U[:, 0]
            scale_col = Vt.t()[0]

            scale = max_singular_value * scale_row.view(-1, 1) @ scale_col.view(1, -1)
            W_q, _ = quantize(W_tile, scale, code)
            W[start_row:end_row, start_col:end_col] = W_q