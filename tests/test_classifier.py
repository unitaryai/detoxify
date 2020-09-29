from pytorch_lightning import Trainer, seed_everything

from src.train import BERTClassifier
import src.data_loaders as module_data
from src.data_loaders import (
    JigsawData,
    JigsawDataBERT,
    JigsawDataBiasBERT,
    JigsawDataMultilingualBERT,
)
from torch.utils.data import DataLoader
from src.utils import ignore_none_collate
import json
import os
import torch


def test_classifier():
    seed_everything(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_instance(module, name, config, *args, **kwargs):
        return getattr(module, config[name]["type"])(
            *args, **config[name]["args"], **kwargs
        )

    config = json.load(
        open("tests/dummy_configs/BERT_toxic_comment_classification.json")
    )

    model = BERTClassifier(config)
    model.to(device)

    dataset = get_instance(module_data, "dataset", config)
    val_dataset = get_instance(module_data, "dataset", config, train=False)

    data_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=12,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=ignore_none_collate,
    )

    valid_data_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=10,
        shuffle=False,
        collate_fn=ignore_none_collate,
    )

    trainer = Trainer(
        gpus=0 if torch.cuda.is_available() else None,
        limit_train_batches=10,
        limit_val_batches=5,
        max_epochs=2,
    )
    trainer.fit(model, data_loader, valid_data_loader)

    results = trainer.test(test_dataloaders=valid_data_loader)
    assert results[0]["test_acc"] > 0.6
