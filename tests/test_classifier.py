from pytorch_lightning import Trainer, seed_everything

from train import ToxicClassifier
import src.data_loaders as module_data
from src.data_loaders import (
    JigsawData,
    JigsawDataOriginal,
    JigsawDataBias,
    JigsawDataMultilingual,
)
from torch.utils.data import DataLoader
import json
import os
import torch


def test_classifier():
    seed_everything(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = json.load(
        open("tests/dummy_configs/BERT_toxic_comment_classification.json")
    )

    def get_instance(module, name, config, *args, **kwargs):
        return getattr(module, config[name]["type"])(
            *args, **config[name]["args"], **kwargs
        )

    model = ToxicClassifier(config)
    model.to(device)

    dataset = get_instance(module_data, "dataset", config)
    val_dataset = get_instance(module_data, "dataset", config, train=False)

    data_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=5,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    valid_data_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=5,
        shuffle=False,
    )

    trainer = Trainer(
        gpus=0 if torch.cuda.is_available() else None,
        limit_train_batches=3,
        limit_val_batches=2,
        max_epochs=1,
    )
    trainer.fit(model, data_loader, valid_data_loader)

    results = trainer.test(test_dataloaders=valid_data_loader)
    print(results)
    assert results > 0.6

if __name__ == "__main__":
    test_classifier()