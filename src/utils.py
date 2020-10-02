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


def pad_item(batch):
    """Pads text items in a batch to the length
    of the longest item in the batch.
    """
    mod_batch = {}
    squeezed_batch = {}
    padded_batch = {}

    for key in batch[0].keys():
        mod_batch[key] = [b[key] for b in batch]
        squeezed_batch[key] = [
            torch.squeeze(mod_batch[key][s]) for s in range(len(mod_batch[key]))
        ]
        padded_batch[key] = pad_sequence(squeezed_batch[key], batch_first=True)
    return padded_batch


def ignore_none_collate(batch):
    """Custom collate function to ignore None types in batch
    and deal with tokenised text data.
    """
    outputs = []
    for i, elem in enumerate(batch[0]):
        if isinstance(elem, dict) and "input_ids" in elem:
            outputs.append(pad_item([b[i] for b in batch if b[0] is not None]))
        else:
            outputs.append(default_collate([b[i] for b in batch if b[0] is not None]))
    return outputs
