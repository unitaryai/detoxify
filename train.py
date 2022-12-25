import argparse
import json
import os

import pytorch_lightning as pl
import src.data_loaders as module_data
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from src.utils import get_model_and_tokenizer
from torch.nn import functional as F
from torch.utils.data import DataLoader


class ToxicClassifier(pl.LightningModule):
    """Toxic comment classification for the Jigsaw challenges.
    Args:
        config ([dict]): takes in args from a predefined config
                              file containing hyperparameters.
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = config["arch"]["args"]["num_classes"]
        self.model_args = config["arch"]["args"]
        self.model, self.tokenizer = get_model_and_tokenizer(**self.model_args)
        self.bias_loss = False

        if "loss_weight" in config:
            self.loss_weight = config["loss_weight"]
        if "num_main_classes" in config:
            self.num_main_classes = config["num_main_classes"]
            self.bias_loss = True
        else:
            self.num_main_classes = self.num_classes

        self.config = config

    def forward(self, x):
        inputs = self.tokenizer(x, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
        outputs = self.model(**inputs)[0]
        return outputs

    def training_step(self, batch, batch_idx):
        x, meta = batch
        output = self.forward(x)
        loss = self.binary_cross_entropy(output, meta)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, meta = batch
        output = self.forward(x)
        loss = self.binary_cross_entropy(output, meta)
        acc = self.binary_accuracy(output, meta)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return {"loss": loss, "acc": acc}

    def test_step(self, batch, batch_idx):
        x, meta = batch
        output = self.forward(x)
        loss = self.binary_cross_entropy(output, meta)
        acc = self.binary_accuracy(output, meta)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return {"loss": loss, "acc": acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), **self.config["optimizer"]["args"])

    def binary_cross_entropy(self, input, meta):
        """Custom binary_cross_entropy function.

        Args:
            output ([torch.tensor]): model predictions
            meta ([dict]): meta dict of tensors including targets and weights

        Returns:
            [torch.tensor]: model loss
        """

        if "weight" in meta:
            target = meta["target"].to(input.device).reshape(input.shape)
            weight = meta["weight"].to(input.device).reshape(input.shape)
            return F.binary_cross_entropy_with_logits(input, target, weight=weight)
        elif "multi_target" in meta:
            target = meta["multi_target"].to(input.device)
            loss_fn = F.binary_cross_entropy_with_logits
            mask = target != -1
            loss = loss_fn(input, target.float(), reduction="none")

            if "class_weights" in meta:
                weights = meta["class_weights"][0].to(input.device)
            elif "weights1" in meta:
                weights = meta["weights1"].to(input.device)
            else:
                weights = torch.tensor(1 / self.num_main_classes).to(input.device)
                loss = loss[:, : self.num_main_classes]
                mask = mask[:, : self.num_main_classes]

            weighted_loss = loss * weights
            nz = torch.sum(mask, 0) != 0
            masked_tensor = weighted_loss * mask
            masked_loss = torch.sum(masked_tensor[:, nz], 0) / torch.sum(mask[:, nz], 0)
            loss = torch.sum(masked_loss)
            return loss
        else:
            target = meta["target"].to(input.device)
            return F.binary_cross_entropy_with_logits(input, target.float())

    def binary_accuracy(self, output, meta):
        """Custom binary_accuracy function.

        Args:
            output ([torch.tensor]): model predictions
            meta ([dict]): meta dict of tensors including targets and weights

        Returns:
            [torch.tensor]: model accuracy
        """
        if "multi_target" in meta:
            target = meta["multi_target"].to(output.device)
        else:
            target = meta["target"].to(output.device)
        with torch.no_grad():
            mask = target != -1
            pred = torch.sigmoid(output[mask]) >= 0.5
            correct = torch.sum(pred.to(output[mask].device) == target[mask])
            if torch.sum(mask).item() != 0:
                correct = correct.item() / torch.sum(mask).item()
            else:
                correct = 0

        return torch.tensor(correct)


def cli_main():
    pl.seed_everything(1234)

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    parser.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: None)",
    )
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="number of workers used in the data loader (default: 10)",
    )
    parser.add_argument("-e", "--n_epochs", default=100, type=int, help="if given, override the num")

    args = parser.parse_args()
    config = json.load(open(args.config))

    if args.device is not None:
        config["device"] = args.device

    # data
    def get_instance(module, name, config, *args, **kwargs):
        return getattr(module, config[name]["type"])(*args, **config[name]["args"], **kwargs)

    dataset = get_instance(module_data, "dataset", config)
    val_dataset = get_instance(module_data, "dataset", config, train=False)

    data_loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    valid_data_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=args.num_workers,
        shuffle=False,
    )
    # model
    model = ToxicClassifier(config)

    # training

    checkpoint_callback = ModelCheckpoint(
        save_top_k=100,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    trainer = pl.Trainer(
        gpus=args.device,
        max_epochs=args.n_epochs,
        accumulate_grad_batches=config["accumulate_grad_batches"],
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=args.resume,
        default_root_dir="saved/" + config["name"],
        deterministic=True,
    )
    trainer.fit(model, data_loader, valid_data_loader)


if __name__ == "__main__":
    cli_main()
