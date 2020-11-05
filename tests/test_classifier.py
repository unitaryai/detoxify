from pytorch_lightning import Trainer, seed_everything

from train import ToxicClassifier
import json
import os
import torch
from run_prediction import predict, load_model


def example_prediction(model_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint=None, device=device, model_type=model_type)
    scores1 = predict(model, "example input")
    scores2 = predict(model, ["example input 1", "example input 2"])
    return scores1, scores2


def test_original_classifier():
    scores1, scores2 = example_prediction("original")
    assert scores1.shape == (1, 6)
    assert scores2.shape == (2, 6)


def test_bias_classifier():
    scores1, scores2 = example_prediction("bias")
    assert scores1.shape == (1, 16)
    assert scores2.shape == (2, 16)


def test_multilingual_classifier():
    scores1, scores2 = example_prediction("multilingual")
    assert scores1.shape == (1, 1)
    assert scores2.shape == (2, 1)
