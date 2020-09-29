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
from src.metric import binary_cross_entropy, binary_accuracy
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

    def forward(self, inputs):
        outputs = self.bert(**inputs)
        return outputs[0]

    def training_step(self, batch, batch_idx):
        x, meta = batch
        output = self(x)
        loss = binary_cross_entropy(output, meta)
        result = pl.TrainResult(minimize=loss)
        result.log("train_loss", loss)
        return result

    def validation_step(self, batch, batch_idx):
        x, meta = batch
        output = self(x)
        loss = binary_cross_entropy(output, meta)
        result = pl.EvalResult(checkpoint_on=loss)
        acc = binary_accuracy(output, meta)
        result.log("val_loss", loss)
        result.log("val_acc", acc)
        return result

    def test_step(self, batch, batch_idx):
        x, meta = batch
        output = self(x)
        loss = F.cross_entropy(output, meta)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log("test_loss", loss)
        result.log("test_acc", accuracy(output, meta))
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), **self.hparams["config"]["optimizer"]["args"]
        )


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
