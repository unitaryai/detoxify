dependencies = ["torch", "transformers"]


def toxic_bert():
    checkpoint = "https://github.com/laurahanu/detoxify/releases/download/v0.1-alpha/toxic_original-4e693588.ckpt"
    loaded = torch.hub.load_state_dict_from_url(checkpoint, progress=False)
    return loaded


def toxic_roberta_bias():
    checkpoint = "https://github.com/laurahanu/detoxify/releases/download/v0.1-alpha/toxic_bias-4e693588.ckpt"
    loaded = torch.hub.load_state_dict_from_url(checkpoint, progress=False)
    return loaded


def toxic_xlmr_multilingual():
    checkpoint = "https://github.com/laurahanu/detoxify/releases/download/v0.1-alpha/toxic_multilingual-bbddc277.ckpt"
    loaded = torch.hub.load_state_dict_from_url(checkpoint, progress=False)
    return loaded