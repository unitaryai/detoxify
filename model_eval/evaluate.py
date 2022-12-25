import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import src.data_loaders as module_data
import torch
from sklearn.metrics import roc_auc_score
from src.data_loaders import JigsawDataBias, JigsawDataMultilingual, JigsawDataOriginal
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import ToxicClassifier


def test_classifier(config, dataset, checkpoint_path, device="cuda:0"):

    model = ToxicClassifier(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)

    def get_instance(module, name, config, *args, **kwargs):
        return getattr(module, config[name]["type"])(*args, **config[name]["args"], **kwargs)

    config["dataset"]["args"]["test_csv_file"] = dataset

    test_dataset = get_instance(module_data, "dataset", config, train=False)

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=int(config["batch_size"]),
        num_workers=20,
        shuffle=False,
    )

    scores = []
    targets = []
    ids = []
    for *items, meta in tqdm(test_data_loader):
        if "multi_target" in meta:
            targets += meta["multi_target"]
        else:
            targets += meta["target"]

        ids += meta["text_id"]
        with torch.no_grad():
            out = model.forward(*items)
            # TODO: save embeddings
            sm = torch.sigmoid(out).cpu().detach().numpy()
        scores.extend(sm)

    binary_scores = [s >= 0.5 for s in scores]
    binary_scores = np.stack(binary_scores)
    scores = np.stack(scores)
    targets = np.stack(targets)
    auc_scores = []

    for class_idx in range(scores.shape[1]):
        mask = targets[:, class_idx] != -1
        target_binary = targets[mask, class_idx]
        class_scores = scores[mask, class_idx]
        try:
            auc = roc_auc_score(target_binary, class_scores)
            auc_scores.append(auc)
        except Exception:
            warnings.warn(
                "Only one class present in y_true. ROC AUC score is not defined in that case. Set to nan for now."
            )
            auc_scores.append(np.nan)

    mean_auc = np.mean(auc_scores)

    results = {
        "scores": scores.tolist(),
        "targets": targets.tolist(),
        "auc_scores": auc_scores,
        "mean_auc": mean_auc,
        "ids": [i.tolist() for i in ids],
    }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-ckpt",
        "--checkpoint",
        type=str,
        help="path to a saved checkpoint",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda:0",
        type=str,
        help="device name e.g., 'cpu' or 'cuda' (default cuda:0)",
    )
    parser.add_argument(
        "-t",
        "--test_csv",
        default=None,
        type=str,
        help="path to test dataset",
    )

    args = parser.parse_args()
    config = json.load(open(args.config))

    if args.device is not None:
        config["gpus"] = args.device

    results = test_classifier(config, args.test_csv, args.checkpoint, args.device)
    test_set_name = args.test_csv.split("/")[-1:][0]

    with open(args.checkpoint[:-4] + f"results_{test_set_name}.json", "w") as f:
        json.dump(results, f)
