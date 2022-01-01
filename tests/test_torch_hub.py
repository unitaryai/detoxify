import gc

import torch


def test_torch_hub_models():
    result = torch.hub.list("unitaryai/detoxify", skip_validation=True)


def test_torch_hub_bert():
    model = torch.hub.load("unitaryai/detoxify", "toxic_bert", skip_validation=True)
    del model
    gc.collect()


def test_torch_hub_roberta():
    model = torch.hub.load("unitaryai/detoxify", "unbiased_toxic_roberta", skip_validation=True)
    del model
    gc.collect()


def test_torch_hub_multilingual():
    model = torch.hub.load("unitaryai/detoxify", "multilingual_toxic_xlm_r", skip_validation=True)
    del model
    gc.collect()
