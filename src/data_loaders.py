from torch.utils.data.dataset import Dataset
import torch
import pandas as pd
import numpy as np
import datasets
from tqdm import tqdm


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
            if isinstance(train_csv_file, list):
                train_set_pd = self.load_data(train_csv_file)
            else:
                train_set_pd = pd.read_csv(train_csv_file)
            self.train_set_pd = train_set_pd
            if "toxicity" not in train_set_pd.columns:
                train_set_pd.rename(columns={"target": "toxicity"}, inplace=True)
            self.train_set = datasets.Dataset.from_pandas(train_set_pd)

        if create_val_set:
            data = self.train_set.train_test_split(val_fraction)
            self.train_set = data["train"]
            self.val_set = data["test"]

        if test_csv_file is not None:
            val_set = pd.read_csv(test_csv_file)
            if add_test_labels:
                data_labels = pd.read_csv(test_csv_file[:-4] + "_labels.csv")
                for category in data_labels.columns[1:]:
                    val_set[category] = data_labels[category]
            val_set = datasets.Dataset.from_pandas(val_set)
            self.val_set = val_set

        if train:
            self.data = self.train_set
        else:
            self.data = self.val_set

        self.train = train

    def __len__(self):
        return len(self.data)

    def load_data(self, train_csv_file):
        files = []
        cols = ["id", "comment_text", "toxic"]
        for file in tqdm(train_csv_file):
            file_df = pd.read_csv(file)
            file_df = file_df[cols]
            file_df = file_df.astype({"id": "string"}, {"toxic": "float64"})
            files.append(file_df)
        train = pd.concat(files)
        return train


class JigsawDataOriginal(JigsawData):
    """Dataloader for the original Jigsaw Toxic Comment Classification Challenge.
    Source: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
    """

    def __init__(
        self,
        train_csv_file="jigsaw_data/train.csv",
        test_csv_file="jigsaw_data/test.csv",
        train=True,
        val_fraction=0.1,
        create_val_set=True,
        add_test_labels=True,
        classes=["toxic"],
    ):

        super().__init__(
            train_csv_file=train_csv_file,
            test_csv_file=test_csv_file,
            train=train,
            val_fraction=val_fraction,
            add_test_labels=add_test_labels,
            create_val_set=create_val_set,
        )
        self.classes = classes

    def __getitem__(self, index):
        meta = {}
        entry = self.data[index]
        text_id = entry["id"]
        text = entry["comment_text"]

        target_dict = {
            label: value for label, value in entry.items() if label in self.classes
        }

        meta["multi_target"] = torch.tensor(
            list(target_dict.values()), dtype=torch.int32
        )
        meta["text_id"] = text_id

        return text, meta


class JigsawDataBias(JigsawData):
    """Dataloader for the Jigsaw Unintended Bias in Toxicity Classification.
    Source: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/
    """

    def __init__(
        self,
        train_csv_file="jigsaw_data/train.csv",
        test_csv_file="jigsaw_data/test.csv",
        train=True,
        val_fraction=0.1,
        create_val_set=True,
        compute_bias_weights=True,
        loss_weight=0.75,
        classes=["toxic"],
        identity_classes=["female"],
    ):

        self.classes = classes

        self.identity_classes = identity_classes

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
        self.loss_weight = loss_weight

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
            {label: 0 for label in identity_target if 0 <= identity_target[label] < 0.5}
        )

        target_dict.update(identity_target)

        meta["multi_target"] = torch.tensor(
            list(target_dict.values()), dtype=torch.float32
        )
        meta["text_id"] = text_id

        if self.train:
            meta["weights"] = self.weights[index]
            toxic_weight = (
                self.weights[index] * self.loss_weight * 1.0 / len(self.classes)
            )
            identity_weight = (1 - self.loss_weight) * 1.0 / len(self.identity_classes)
            meta["weights1"] = torch.tensor(
                [
                    *[toxic_weight] * len(self.classes),
                    *[identity_weight] * len(self.identity_classes),
                ]
            )

        return text, meta

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


class JigsawDataMultilingual(JigsawData):
    """Dataloader for the Jigsaw Multilingual Toxic Comment Classification.
    Source: https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/
    """

    def __init__(
        self,
        train_csv_file="jigsaw_data/multilingual_challenge/jigsaw-toxic-comment-train.csv",
        test_csv_file="jigsaw_data/multilingual_challenge/validation.csv",
        train=True,
        val_fraction=0.1,
        create_val_set=False,
        classes=["toxic"],
    ):

        self.classes = classes
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
        if "translated" in entry:
            text = entry["translated"]
        elif "comment_text_en" in entry:
            text = entry["comment_text_en"]
        else:
            text = entry["comment_text"]

        target_dict = {label: 1 if entry[label] >= 0.5 else 0 for label in self.classes}
        meta["target"] = torch.tensor(list(target_dict.values()), dtype=torch.int32)
        meta["text_id"] = text_id

        return text, meta
