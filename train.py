from argparse import ArgumentParser
import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from torch.nn import functional as F
import transformers
from torch.utils.data import DataLoader
import json
import argparse
import src.data_loaders as module_data

from src.utils import move_to, ignore_none_collate
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    PretrainedConfig,
    BertConfig,
)
from src.data_loaders import (
    JigsawData,
    JigsawDataBERT,
    JigsawDataBiasBERT,
    JigsawDataMultilingualBERT,
)


class BERTClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        configuration = BertConfig()
        self.num_classes = config["arch"]["args"]["num_classes"]
        configuration.num_labels = self.num_classes

        self.bert = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", config=configuration
        )

        self.new_fc_layers = [self.bert.classifier]

        if config["arch"]["args"]["freeze"]:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.bias_loss = False

        if "loss_weight" in config:
            self.loss_weight = config["loss_weight"]
        if "num_main_classes" in config:
            self.num_main_classes = config["num_main_classes"]
            self.bias_loss = True

    def forward(self, inputs):
        outputs = self.bert(**inputs)
        return outputs[0]

    def training_step(self, batch, batch_idx):
        x, meta = batch
        output = self(x)
        loss = self.binary_cross_entropy(output, meta)
        result = pl.TrainResult(minimize=loss)
        result.log("train_loss", loss)
        return result

    def validation_step(self, batch, batch_idx):
        x, meta = batch
        output = self(x)
        loss = self.binary_cross_entropy(output, meta)
        result = pl.EvalResult(checkpoint_on=loss)
        acc = self.binary_accuracy(output, meta)
        result.log("val_loss", loss)
        result.log("val_acc", acc)
        return result

    def test_step(self, batch, batch_idx):
        x, meta = batch
        output = self(x)
        loss = self.binary_cross_entropy(output, meta)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log("test_loss", loss)
        result.log("test_acc", self.binary_accuracy(output, meta))
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), **self.hparams["config"]["optimizer"]["args"]
        )

    def binary_cross_entropy(self, input, meta):

        if "multi_target" in meta:
            target = meta["multi_target"].to(input.device)
            loss_fn = F.binary_cross_entropy_with_logits
            loss = 0
            identity_loss = 0

        if "weights" in meta:
            meta["weights"].to(input.device)
        else:
            meta["weights"] = torch.ones(target.shape[0]).to(input.device)

        if not self.bias_loss:
            self.num_main_classes = self.num_classes
            self.loss_weight = 1

        mask = target != -1
        weight = torch.stack(self.num_main_classes * [meta["weights"].squeeze()], -1)
        loss = loss_fn(
            input,
            target.float(),
            reduction="none",
        )
        loss[:, : self.num_main_classes] = loss[:, : self.num_main_classes] * weight

        loss = loss * mask
        final_loss = torch.sum(loss[:, : self.num_main_classes]) / torch.sum(
            mask[:, : self.num_main_classes]
        )

        if self.bias_loss and torch.sum(mask[:, self.num_main_classes :]) > 0:
            identity_loss = torch.sum(loss[:, self.num_main_classes :]) / torch.sum(
                mask[:, self.num_main_classes :]
            )
        else:
            identity_loss = 0
        loss = final_loss * self.loss_weight + identity_loss * (1 - self.loss_weight)

        return loss

    def binary_accuracy(self, output, meta):
        if "multi_target" in meta:
            correct = 0
            target = meta["multi_target"].to(output.device)
            cnt = 0
            with torch.no_grad():
                for i in range(target.shape[-1]):
                    mask = target[:, i] != -1
                    pred = torch.sigmoid(output[mask, i]) >= 0.5
                    correct = (
                        correct
                        + torch.sum(pred.to(output.device) == target[mask, i]).item()
                    )
                    cnt = cnt + torch.sum(mask)
            return torch.tensor(correct / cnt.item())
        else:
            target = meta["target"].to(output.device).reshape(output.shape) >= 0.5
            with torch.no_grad():
                pred = torch.sigmoid(output) >= 0.5
                correct = torch.sum(pred == target).item()
            return torch.tensor(correct / len(target))


def cli_main():
    pl.seed_everything(1234)

    # args
    parser = argparse.ArgumentParser(description="PyTorch Template")
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
        help="indices of GPUs to enable (default: all)",
    )
    parser.add_argument(
        "-g", "--n_gpu", default=None, type=int, help="if given, override the num"
    )
    parser.add_argument(
        "-e", "--n_epochs", default=100, type=int, help="if given, override the num"
    )

    args = parser.parse_args()
    config = json.load(open(args.config))

    if args.device is not None:
        config["device"] = args.device

    # data
    def get_instance(module, name, config, *args, **kwargs):
        return getattr(module, config[name]["type"])(
            *args, **config[name]["args"], **kwargs
        )

    dataset = get_instance(module_data, "dataset", config)
    val_dataset = get_instance(module_data, "dataset", config, train=False)

    data_loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        num_workers=20,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=ignore_none_collate,
    )

    valid_data_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=20,
        shuffle=False,
        collate_fn=ignore_none_collate,
    )
    # model
    model = BERTClassifier(config)

    # training

    checkpoint_callback = ModelCheckpoint(
        save_top_k=20,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    trainer = pl.Trainer(
        gpus=args.device,
        max_epochs=args.n_epochs,
        accumulate_grad_batches=config["accumulate_grad_batches"],
        checkpoint_callback=checkpoint_callback,
    )
    trainer.fit(model, data_loader, valid_data_loader)

    # trainer.test(test_dataloaders=test_dataset)


if __name__ == "__main__":
    cli_main()
