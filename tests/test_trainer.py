from pytorch_lightning import Trainer, seed_everything

from train import ToxicClassifier
import src.data_loaders as module_data
from torch.utils.data import DataLoader
import json
import torch


def initialize_trainer(CONFIG):
    seed_everything(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_instance(module, name, config, *args, **kwargs):
        return getattr(module, config[name]["type"])(
            *args, **config[name]["args"], **kwargs
        )

    model = ToxicClassifier(CONFIG)
    model.to(device)

    dataset = get_instance(module_data, "dataset", CONFIG)
    val_dataset = get_instance(module_data, "dataset", CONFIG, train=False)

    data_loader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        num_workers=2,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    valid_data_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        num_workers=2,
        shuffle=False,
    )

    trainer = Trainer(
        gpus=0 if torch.cuda.is_available() else None,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
    )
    trainer.fit(model, data_loader, valid_data_loader)
    results = trainer.test(test_dataloaders=valid_data_loader)

    return results


def test_trainer():
    CONFIG = json.load(open("configs/Toxic_comment_classification_BERT.json"))
    CONFIG["dataset"]["args"][
        "train_csv_file"
    ] = "tests/dummy_data/jigsaw-toxic-comment-classification-challenge/train.csv"
    CONFIG["dataset"]["args"][
        "test_csv_file"
    ] = "tests/dummy_data/jigsaw-toxic-comment-classification-challenge/test.csv"
    CONFIG["batch_size"] = 2

    results = initialize_trainer(CONFIG)
    print(results)
    assert results[0]["test_acc"] > 0.6


if __name__ == "__main__":
    test_trainer()
