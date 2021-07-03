

from transformers import (
    BertForSequenceClassification,
    AlbertForSequenceClassification,
    RobertaForSequenceClassification,
    XLMRobertaForSequenceClassification
)

from detoxify.detoxify import (
    toxic_bert, # "original"
    toxic_albert, # "original-small"
    unbiased_toxic_roberta, # "unbiased"
    unbiased_albert, # "unbiased-small"
    multilingual_toxic_xlm_r # # "multilingual"

)

def test_model_toxic_bert():
    model = toxic_bert()
    assert isinstance(model, BertForSequenceClassification)

def test_model_toxic_albert():
    model = toxic_albert()
    assert isinstance(model, AlbertForSequenceClassification)

def test_model_unbiased_toxic_roberta():
    model = unbiased_toxic_roberta()
    assert isinstance(model, RobertaForSequenceClassification)

def test_model_unbiased_albert():
    model = unbiased_albert()
    assert isinstance(model, AlbertForSequenceClassification)

def test_model_multilingual_toxic_xlm_r():
    model = multilingual_toxic_xlm_r()
    assert isinstance(model, XLMRobertaForSequenceClassification)
