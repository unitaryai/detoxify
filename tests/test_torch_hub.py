import torch
import gc

def test_torch_hub_models():
    result = torch.hub.list("unitaryai/detoxify")

def test_torch_hub_bert():
    model = torch.hub.load('unitaryai/detoxify', 'toxic_bert')
    del model
    gc.collect()

def test_torch_hub_roberta():
    model = torch.hub.load('unitaryai/detoxify', 'unbiased_toxic_roberta')
    del model
    gc.collect()

def test_torch_hub_multilingual():
    model = torch.hub.load('unitaryai/detoxify', 'multilingual_toxic_xlm_r')
    del model
    gc.collect()
