import torch
from torch.utils.data._utils.collate import default_collate
from torch.nn.utils.rnn import pad_sequence


def move_to(obj, device):
    """Function to move objects of different types
    containing a tensor to device.
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")
