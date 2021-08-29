import torch

def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to_device(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to_device(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to_device")

def squeeze_dim(obj, dim):
    if torch.is_tensor(obj):
        return torch.squeeze(obj, dim=dim)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = squeeze_dim(v, dim)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(squeeze_dim(v, dim))
        return res
    else:
        raise TypeError("Invalid type for move_to_device")