import torch
import sys

# ==============af4 int===========================
def create_afint_numbers(xmax, xmin):
    result = []
    for i in range(len(xmax)):
        pos = xmax[i]
        neg = xmin[i]
        if abs(pos) > abs(neg):
            pos_points = torch.linspace(pos, 0, 9)
            neg_points = torch.linspace(neg, 0, 8)[:-1]
        else:
            pos_points = torch.linspace(pos, 0, 8)
            neg_points = torch.linspace(neg, 0, 9)[:-1]
        result.append(pos_points)
        result.append(neg_points)
    result = torch.cat(result)
    return result



# ==============af4 fp===========================
def create_affp_numbers(xmax, xmin):
    result = []
    for i in range(len(xmax)):
        pos_scale = xmax[i] / 6
        neg_scale = xmin[i] / 6
        code_affp = torch.tensor([6*pos_scale, 4*pos_scale, 3*pos_scale, 2*pos_scale, 1.5*pos_scale, 1*pos_scale, 0.5*pos_scale, 0, 0, 0.5*neg_scale, 1*neg_scale, 1.5*neg_scale, 2*neg_scale, 3*neg_scale, 4*neg_scale, 6*neg_scale])
        result.append(code_affp)
    result = torch.cat(result)
    return result




# # ===========shijie=============================

# def create_quantiles(tensor, nquantile):
#     #Check if all tensor values are positive  
#     #if not torch.all(tensor > 0):  
#     #    raise ValueError("All values in the tensor must be positive.")  
#     # Flatten the tensor to 1D  
#     #flat_tensor = tensor.view(-1)  
  
#     # Count the number of elements  
#     num_elements = tensor.nelement()  
#     # Sort the tensor values in ascending order  
#     sorted_tensor = torch.sort(tensor).values  
#     # Calculate the step  
#     step = num_elements // nquantile
#     # Get the numbers at index step, 2*step, 3*step, ...  
#     quantile_values = sorted_tensor[step-1::step]
#     # in case the length of nquantile_values is more than nquantile
#     quantile_values = quantile_values[:nquantile]
#     return quantile_values 


# def create_af_numbers(tensor, bits):
#     N = 2
#     if bits == 4:
#         num_quantiles_large = 8 + N
#         num_quantiles_small = 7 + N
#     elif bits == 3:
#         num_quantiles_large = 4
#         num_quantiles_small = 3
#     else:
#         raise ValueError("Bit width not supported.")

#     # Flatten the tensor to 1D  
#     flat_tensor = tensor.view(-1)  
  
#     # Get all the positive numbers in the tensor  
#     positive_numbers = flat_tensor[flat_tensor > 0]  
#     positive_nelement = positive_numbers.nelement()

#     # Get all the negative numbers in the tensor  
#     negative_numbers = torch.abs(flat_tensor[flat_tensor < 0])
#     negative_nelement = negative_numbers.nelement()
    
#     if positive_nelement > negative_nelement:
#         positive_quantile_values = create_quantiles(positive_numbers,num_quantiles_large)
#         negative_quantile_values = create_quantiles(negative_numbers,num_quantiles_small)
#     else:
#         positive_quantile_values = create_quantiles(positive_numbers,num_quantiles_small)
#         negative_quantile_values = create_quantiles(negative_numbers,num_quantiles_large)

#     #reverse and opposite    
#     negative_quantile_values = -torch.flip(negative_quantile_values, dims=(0,))
#     device = negative_quantile_values.device
#     #real quantiles
#     # quantile_values = torch.cat((negative_quantile_values,torch.zeros(1,device=device),positive_quantile_values),dim=0)
#     quantile_values = torch.cat((negative_quantile_values[:-N],torch.zeros(1,device=device),positive_quantile_values[N:]),dim=0)
#     #normalized quantiles
#     # positive_scale = positive_quantile_values[-1]
#     # negative_scale = -negative_quantile_values[0]
#     # normalized_quantile_values = torch.cat((negative_quantile_values/negative_scale,torch.zeros(1,device=device),positive_quantile_values/positive_scale),dim=0)
#     # return quantile_values,normalized_quantile_values,negative_scale,positive_scale
#     return quantile_values

# # random_tensor = torch.randn(100)
# # print(random_tensor)
# # quantile_values,normalized_quantile_values,negative_scale,positive_scale = create_af_numbers(random_tensor,4)
# # print(quantile_values,normalized_quantile_values,negative_scale,positive_scale)

# # ==========================================