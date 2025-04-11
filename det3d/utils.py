import torch
import numpy as np

# Convert outputs to CPU
def move_to_cpu(item):
    if isinstance(item, torch.Tensor):
        return item.cpu().numpy()
    elif isinstance(item, dict):
        return {k: move_to_cpu(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [move_to_cpu(i) for i in item]
    else:
        return item


def print_dict_tensors_size(tensor_dict):
    if isinstance(tensor_dict, dict):
        for k,v in tensor_dict.items():
            if isinstance(v,torch.Tensor):
                print(f'Size of {k}: {list(v.size())}')
            elif isinstance(v,np.ndarray):
                print(f'Size of {k}: {list(v.shape)}')
    elif isinstance(tensor_dict,torch.Tensor):
        print(f'Size of the tensor: {list(tensor_dict.size())}')
    
    elif isinstance(tensor_dict,np.ndarray):
        print(f'Size of the ndarray: {list(tensor_dict.shape)}')

def move_to_gpu(item):
    if isinstance(item, torch.Tensor):
        return item.float().cuda()
    elif isinstance(item, np.ndarray):
        return torch.from_numpy(item).float().cuda()
    elif isinstance(item, dict):
        return {k: move_to_gpu(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [move_to_gpu(i) for i in item]
    else:
        return item
    
def load_dict_to_gpu(batch_dict:dict):
    for key, val in batch_dict.items():
        if isinstance(val, np.ndarray):
            batch_dict[key] = torch.from_numpy(val).float().cuda()
        elif isinstance(val,torch.Tensor):
            batch_dict[key] = val.float().cuda()