from detoxify.detoxify import (
    Detoxify,
    multilingual_toxic_xlm_r,
    toxic_albert,
    toxic_bert,
    unbiased_albert,
    unbiased_toxic_roberta,
)
from transformers import (
    AlbertForSequenceClassification,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    XLMRobertaForSequenceClassification,
)

CLASSES = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]


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


def test_original():
    model = Detoxify("original")
    results = model.predict(["shut up, you liar", "i am a jewish woman who is blind"])
    assert len(results) == 6
    assert all(cl in results for cl in CLASSES[:6])
    assert results["toxicity"][0] >= 0.7
    assert results["toxicity"][1] < 0.5


def test_original_small():
    model = Detoxify("original-small")
    results = model.predict(["shut up, you liar", "i am a jewish woman who is blind"])
    assert len(results) == 6
    assert all(cl in results for cl in CLASSES[:6])
    assert results["toxicity"][0] >= 0.7
    assert results["toxicity"][1] < 0.5


def test_unbiased_model():
    model = Detoxify("unbiased")
    results = model.predict(["shut up, you liar", "i am a jewish woman who is blind"])
    assert len(results) == 7
    assert all(cl in results for cl in CLASSES)
    assert results["toxicity"][0] >= 0.7
    assert results["toxicity"][1] < 0.5


def test_unbiased_small():
    model = Detoxify("unbiased-small")
    results = model.predict(["shut up, you liar", "i am a jewish woman who is blind"])
    assert len(results) == 7
    assert all(cl in results for cl in CLASSES)
    assert results["toxicity"][0] >= 0.7
    assert results["toxicity"][1] < 0.5


def test_multilingual():
    model = Detoxify("multilingual")
    results = model.predict(["tais toi, tu es un menteur", "ben kör bir yahudi kadınıyım"])
    assert len(results) == 7
    assert all(cl in results for cl in CLASSES)
    assert results["toxicity"][0] >= 0.7
    assert results["toxicity"][1] < 0.5
