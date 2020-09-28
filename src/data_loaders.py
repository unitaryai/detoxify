from torch.utils.data.dataset import Dataset
import torch
import os
import pandas as pd
import numpy as np
from transformers import BertTokenizer
import torch


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
        train_fraction=0.9,
        add_test_labels=False,
        create_val_set=True,
    ):

        if train_csv_file is not None:
            train_set = pd.read_csv(train_csv_file)

        if create_val_set:
            train_set, val_set = self.create_validation_set(
                train_set, train_fraction=train_fraction
            )
        if test_csv_file is not None:
            val_set = pd.read_csv(test_csv_file)
            if add_test_labels:
                data_labels = pd.read_csv(test_csv_file[:-4] + "_labels.csv")
                for category in data_labels.columns[1:]:
                    val_set[category] = data_labels[category]
                    val_set.drop(
                        val_set.loc[val_set["toxic"] == -1].index, inplace=True
                    )

        if train:
            data = train_set
        else:
            data = val_set

        self.data = data

    def __len__(self):
        return len(self.data)

    def create_validation_set(self, train_data, train_fraction=0.9):
        np.random.seed(0)
        indices = np.random.rand(len(train_data)) < train_fraction
        train_set = train_data[indices]
        val_set = train_data[~indices]
        return train_set, val_set


class JigsawDataBERT(JigsawData):
    """Dataloader for the original Jigsaw Toxic Comment Classification Challenge.
    Source: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
    """

    def __init__(
        self,
        train_csv_file="/data/unitarybot/jigsaw_data/train.csv",
        test_csv_file="/data/unitarybot/jigsaw_data/test.csv",
        train=True,
        train_fraction=0.9,
        create_val_set=True,
        add_test_labels=True,
    ):

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        super().__init__(
            train_csv_file=train_csv_file,
            test_csv_file=test_csv_file,
            train=train,
            train_fraction=train_fraction,
            add_test_labels=add_test_labels,
            create_val_set=create_val_set,
        )
        self.classes = list(self.data.columns[2:])

    def __getitem__(self, index):
        meta = {}
        pd_index = self.data.index[index]
        text_id = self.data.id[pd_index]
        text = self.data.comment_text[pd_index]
        target_dict = dict(self.data.iloc[index][2:])

        tokenised_text = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        tokenised_text = tokenised_text.data
        meta["multi_target"] = torch.tensor(
            list(target_dict.values()), dtype=torch.int32
        )
        meta["text_id"] = text_id

        return tokenised_text, meta


class JigsawDataBERT_unintended_bias(JigsawData):
    """Dataloader for the Jigsaw Unintended Bias in Toxicity Classification.
    Source: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/
    """

    def __init__(
        self,
        train_csv_file="/data/unitarybot/jigsaw_data/train.csv",
        test_csv_file="/data/unitarybot/jigsaw_data/test.csv",
        train=True,
        train_fraction=0.9,
        create_val_set=True,
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
            train_fraction=train_fraction,
            create_val_set=create_val_set,
        )
        if train:
            self.weights = self.compute_weigths(self.data)
        self.train = train

        if "toxicity" not in self.data.columns:
            self.data.rename(columns={"target": "toxicity"}, inplace=True)

    def __getitem__(self, index):
        meta = {}
        pd_index = self.data.index[index]
        text_id = self.data.id[pd_index]
        text = self.data.comment_text[pd_index]
        target_dict = {
            l: 1 if self.data.iloc[index][l] >= 0.5 else 0 for l in self.classes
        }

        identity_target = {
            l: -1 if np.isnan(self.data.iloc[index][l]) else self.data.iloc[index][l]
            for l in self.identity_classes
        }
        identity_target.update(
            {l: 1 for l in identity_target if identity_target[l] >= 0.5}
        )
        identity_target.update(
            {l: 0 for l in identity_target if 0 < identity_target[l] < 0.5}
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

        return tokenised_text, meta

    def compute_weigths(self, train_df):
        """Inspired from 2nd solution.
        Source: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/100661"""

        subgroup_bool = (train_df[self.identity_classes].fillna(0) >= 0.5).sum(
            axis=1
        ) > 0
        positive_bool = train_df["target"] >= 0.5
        weights = np.ones(len(train_df)) * 0.25

        # Backgroud Positive and Subgroup Negative
        weights[
            ((~subgroup_bool) & (positive_bool)) | ((subgroup_bool) & (~positive_bool))
        ] += 0.25
        weights[(subgroup_bool)] += 0.25
        return weights


class JigsawDataBERT_multilingual_challenge(JigsawData):
    """Dataloader for the Jigsaw Multilingual Toxic Comment Classification.
    Source: https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/
    """

    def __init__(
        self,
        train_csv_file="/data/unitarybot/jigsaw_data/multilingual_challenge/jigsaw-toxic-comment-train.csv",
        test_csv_file="/data/unitarybot/jigsaw_data/multilingual_challenge/validation.csv",
        train=True,
        train_fraction=0.9,
        create_val_set=False,
    ):

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.classes = ["toxic"]
        super().__init__(
            train_csv_file=train_csv_file,
            test_csv_file=test_csv_file,
            train=train,
            train_fraction=train_fraction,
            create_val_set=create_val_set,
        )

    def __getitem__(self, index):
        meta = {}
        pd_index = self.data.index[index]
        text_id = self.data.id[pd_index]
        text = self.data.comment_text[pd_index]
        target_dict = {
            l: 1 if self.data.iloc[index][l] >= 0.5 else 0 for l in self.classes
        }

        tokenised_text = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        tokenised_text = tokenised_text.data
        meta["target"] = torch.tensor(list(target_dict.values()), dtype=torch.int32)
        meta["text_id"] = text_id

        return tokenised_text, meta