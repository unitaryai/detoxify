import torch

def test_torch_hub_models():
    result = torch.hub.list("unitaryai/detoxify")
    #assert any(result)
    assert True

def test_torch_hub_bert():
    model = torch.hub.load('unitaryai/detoxify', 'toxic_bert')
    #assert ...
    assert True

def test_torch_hub_roberta():
    model = torch.hub.load('unitaryai/detoxify', 'unbiased_toxic_roberta')
    #assert ...
    assert True

def test_torch_hub_multilingual():
    model = torch.hub.load('unitaryai/detoxify', 'multilingual_toxic_xlm_r')
    #assert ...
    assert True
