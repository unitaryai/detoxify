import torch
from torch.utils.data._utils.collate import default_collate
from torch.nn.utils.rnn import pad_sequence
import transformers


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


def get_model_and_tokenizer(model_type, model_name, tokenizer_name, num_classes):
    model = getattr(transformers, model_name).from_pretrained(
        model_type, num_labels=num_classes
    )
    tokenizer = getattr(transformers, tokenizer_name).from_pretrained(model_type)

    return model, tokenizer