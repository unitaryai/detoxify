from torch.utils.data.dataset import Dataset
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer
import datasets
from datasets import list_datasets, load_dataset, list_metrics, load_metric


class JigsawData(Dataset):
    """Dataloader for the Jigsaw Toxic Comment Classification Challenges.
    If test_csv_file is None and create_val_set is True the train file
    specified gets split into a train and validation set according to
    train_fraction."""

    def __init__(
        self,
        train_csv_file,
        test_csv_file,
        train=True,
        val_fraction=0.9,
        add_test_labels=False,
        create_val_set=True,
    ):

        if train_csv_file is not None:
            train_set_pd = pd.read_csv(train_csv_file)
            self.train_set_pd = train_set_pd
            if "toxicity" not in train_set_pd.columns:
                train_set_pd.rename(columns={"target": "toxicity"}, inplace=True)
            train_set = datasets.Dataset.from_pandas(train_set_pd)

        if create_val_set:
            data = train_set.train_test_split(val_fraction)
            self.train_set = data["train"]
            self.val_set = data["test"]

        if test_csv_file is not None:
            val_set = pd.read_csv(test_csv_file)
            if add_test_labels:
                data_labels = pd.read_csv(test_csv_file[:-4] + "_labels.csv")
                for category in data_labels.columns[1:]:
                    val_set[category] = data_labels[category]
            self.val_set = datasets.Dataset.from_pandas(val_set)

        if train:
            self.data = self.train_set
        else:
            self.data = self.val_set

        self.train = train

    def __len__(self):
        return len(self.data)


class JigsawDataBERT(JigsawData):
    """Dataloader for the original Jigsaw Toxic Comment Classification Challenge.
    Source: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
    """

    def __init__(
        self,
        train_csv_file="/data/unitarybot/jigsaw_data/train.csv",
        test_csv_file="/data/unitarybot/jigsaw_data/test.csv",
        train=True,
        val_fraction=0.1,
        create_val_set=True,
        add_test_labels=True,
    ):

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        super().__init__(
            train_csv_file=train_csv_file,
            test_csv_file=test_csv_file,
            train=train,
            val_fraction=val_fraction,
            add_test_labels=add_test_labels,
            create_val_set=create_val_set,
        )
        self.classes = list(self.data.column_names[2:])

    def __getitem__(self, index):
        meta = {}
        entry = self.data[index]
        text_id = entry["id"]
        text = entry["comment_text"]
        target_dict = {
            label: value for label, value in entry.items() if label in self.classes
        }

        tokenised_text = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        tokenised_text = tokenised_text.data
        meta["multi_target"] = torch.tensor(
            list(target_dict.values()), dtype=torch.int32
        )
        meta["text_id"] = text_id

        return tokenised_text, meta


class JigsawDataBiasBERT(JigsawData):
    """Dataloader for the Jigsaw Unintended Bias in Toxicity Classification.
    Source: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/
    """

    def __init__(
        self,
        train_csv_file="/data/unitarybot/jigsaw_data/train.csv",
        test_csv_file="/data/unitarybot/jigsaw_data/test.csv",
        train=True,
        val_fraction=0.1,
        create_val_set=True,
        compute_bias_weights=True,
    ):

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.classes = [
            "toxicity",
            "severe_toxicity",
            "obscene",
            "identity_attack",
            "insult",
            "threat",
            "sexual_explicit",
        ]
        self.identity_classes = [
            "male",
            "female",
            "homosexual_gay_or_lesbian",
            "christian",
            "jewish",
            "muslim",
            "black",
            "white",
            "psychiatric_or_mental_illness",
        ]

        super().__init__(
            train_csv_file=train_csv_file,
            test_csv_file=test_csv_file,
            train=train,
            val_fraction=val_fraction,
            create_val_set=create_val_set,
        )
        if train:
            if compute_bias_weights:
                self.weights = self.compute_weigths(self.train_set_pd)
            else:
                self.weights = None

        self.train = train

    def __getitem__(self, index):
        meta = {}
        entry = self.data[index]
        text_id = entry["id"]
        text = entry["comment_text"]
        target_dict = {label: 1 if entry[label] >= 0.5 else 0 for label in self.classes}

        identity_target = {
            label: -1 if entry[label] is None else entry[label]
            for label in self.identity_classes
        }
        identity_target.update(
            {label: 1 for label in identity_target if identity_target[label] >= 0.5}
        )
        identity_target.update(
            {label: 0 for label in identity_target if 0 < identity_target[label] < 0.5}
        )
        target_dict.update(identity_target)

        tokenised_text = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        tokenised_text = tokenised_text.data
        meta["multi_target"] = torch.tensor(
            list(target_dict.values()), dtype=torch.int32
        )
        meta["text_id"] = text_id
        if self.train:
            meta["weights"] = self.weights[index]
        else:
            meta["weights"] = torch.tensor([1], dtype=torch.int32)

        meta["toxicity_ids"] = torch.tensor(
            [True] * len(self.classes) + [False] * len(self.identity_classes)
        )
        return tokenised_text, meta

    def compute_weigths(self, train_df):
        """Inspired from 2nd solution.
        Source: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/100661"""
        subgroup_bool = (train_df[self.identity_classes].fillna(0) >= 0.5).sum(
            axis=1
        ) > 0
        positive_bool = train_df["toxicity"] >= 0.5
        weights = np.ones(len(train_df)) * 0.25

        # Backgroud Positive and Subgroup Negative
        weights[
            ((~subgroup_bool) & (positive_bool)) | ((subgroup_bool) & (~positive_bool))
        ] += 0.25
        weights[(subgroup_bool)] += 0.25
        return weights


class JigsawDataMultilingualBERT(JigsawData):
    """Dataloader for the Jigsaw Multilingual Toxic Comment Classification.
    Source: https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/
    """

    def __init__(
        self,
        train_csv_file="/data/unitarybot/jigsaw_data/multilingual_challenge/jigsaw-toxic-comment-train.csv",
        test_csv_file="/data/unitarybot/jigsaw_data/multilingual_challenge/validation.csv",
        train=True,
        val_fraction=0.1,
        create_val_set=False,
    ):

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.classes = ["toxic"]
        super().__init__(
            train_csv_file=train_csv_file,
            test_csv_file=test_csv_file,
            train=train,
            val_fraction=val_fraction,
            create_val_set=create_val_set,
        )

    def __getitem__(self, index):
        meta = {}
        entry = self.data[index]
        text_id = entry["id"]
        text = entry["comment_text"]
        target_dict = {label: 1 if entry[label] >= 0.5 else 0 for label in self.classes}

        tokenised_text = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        tokenised_text = tokenised_text.data
        meta["target"] = torch.tensor(list(target_dict.values()), dtype=torch.int32)
        meta["text_id"] = text_id

        return tokenised_text, meta
