import datasets
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


class JigsawData(Dataset):
    """Base dataloader for the Jigsaw Toxic Comment Classification Challenges."""

    def __init__(self, train_csv_file, test_csv_file, train=True, add_test_labels=False):
        if train:
            self.train_set_pd = self.load_data(train_csv_file)
            self.train_set = datasets.Dataset.from_pandas(self.train_set_pd)
            self.data = self.train_set
        else:
            self.data = self.load_val(test_csv_file, add_labels=add_test_labels)

        self.train = train

    def __len__(self):
        return len(self.data)

    def load_data(self, csv_file):
        change_names = {
            "target": "toxicity",
            "toxic": "toxicity",
            "identity_hate": "identity_attack",
            "severe_toxic": "severe_toxicity",
        }
        if isinstance(csv_file, list):
            files = []
            for file in tqdm(csv_file):
                chunks = []
                for chunk in pd.read_csv(file, chunksize=100000):
                    chunks.append(chunk)

                file_df = pd.concat(chunks, axis=0)
                filtered_change_names = {k: v for k, v in change_names.items() if k in file_df.columns}
                if len(filtered_change_names) > 0:
                    file_df.rename(columns=filtered_change_names, inplace=True)
                file_df = file_df.astype({"id": "string"})
                files.append(file_df)

            final_df = pd.concat(files, join="outer")
        elif isinstance(csv_file, str):
            final_df = pd.read_csv(csv_file)
            filtered_change_names = {k: v for k, v in change_names.items() if k in final_df.columns}
            if len(filtered_change_names) > 0:
                final_df.rename(columns=filtered_change_names, inplace=True)
        return final_df

    def load_val(self, test_csv_file, add_labels=False):
        val_set = self.load_data(test_csv_file)
        if add_labels:
            data_labels = pd.read_csv(test_csv_file[:-4] + "_labels.csv")
            for category in data_labels.columns[1:]:
                val_set[category] = data_labels[category]
        val_set = datasets.Dataset.from_pandas(val_set)
        return val_set

    def filter_entry_labels(self, entry, classes, threshold=0.5, soft_labels=False):
        target = {label: -1 if label not in entry or entry[label] is None else entry[label] for label in classes}
        if not soft_labels:
            target.update({label: 1 for label in target if target[label] >= threshold})
            target.update({label: 0 for label in target if 0 <= target[label] < threshold})
        return target


class JigsawDataOriginal(JigsawData):
    """Dataloader for the original Jigsaw Toxic Comment Classification Challenge.
    Source: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
    """

    def __init__(
        self,
        train_csv_file="jigsaw_data/train.csv",
        test_csv_file="jigsaw_data/test.csv",
        train=True,
        add_test_labels=True,
        classes=["toxic"],
    ):

        super().__init__(
            train_csv_file=train_csv_file,
            test_csv_file=test_csv_file,
            train=train,
            add_test_labels=add_test_labels,
        )
        self.classes = classes

    def __getitem__(self, index):
        meta = {}
        entry = self.data[index]
        text_id = entry["id"]
        text = entry["comment_text"]

        target_dict = {label: value for label, value in entry.items() if label in self.classes}

        meta["multi_target"] = torch.tensor(list(target_dict.values()), dtype=torch.int32)
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
        compute_bias_weights=True,
        loss_weight=0.75,
        classes=["toxic"],
        identity_classes=["female"],
        soft_labels=False,
    ):

        self.classes = classes
        self.soft_labels = soft_labels
        self.identity_classes = identity_classes

        super().__init__(train_csv_file=train_csv_file, test_csv_file=test_csv_file, train=train)
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

        if "translated" in entry:
            text = entry["translated"]
        elif "comment_text_en" in entry:
            text = entry["comment_text_en"]
        else:
            text = entry["comment_text"]

        target_dict = self.filter_entry_labels(
            entry,
            self.classes + self.identity_classes,
            threshold=0.5,
            soft_labels=self.soft_labels,
        )
        meta["multi_target"] = torch.tensor(list(target_dict.values()), dtype=torch.float32)
        meta["text_id"] = text_id

        if self.train:
            meta["weights"] = self.weights[index]
            toxic_weight = self.weights[index] * self.loss_weight * 1.0 / len(self.classes)
            identity_weight = (1 - self.loss_weight) * 1.0 / len(self.identity_classes)
            meta["weights1"] = torch.tensor(
                [
                    *[toxic_weight] * len(self.classes),
                    *[identity_weight] * len(self.identity_classes),
                ]
            )

        return text, meta

    def compute_weigths(self, train_df):
        """Inspired from 2nd best solution.
        Source: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/100661"""
        subgroup_bool = (train_df[self.identity_classes].fillna(0) >= 0.5).sum(axis=1) > 0
        positive_bool = train_df["toxicity"] >= 0.5
        weights = np.ones(len(train_df)) * 0.25

        # Background Positive and Subgroup Negative
        # i.e. weigh higher toxic comments that don't mention identity and non toxic ones that mention it
        weights[((~subgroup_bool) & (positive_bool)) | ((subgroup_bool) & (~positive_bool))] += 0.25
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
        classes=["toxic"],
    ):

        self.classes = classes
        super().__init__(train_csv_file=train_csv_file, test_csv_file=test_csv_file, train=train)

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
