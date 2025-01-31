import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("preprocessing_utils")
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def update_test(test_csv_file):
    """Combines disjointed test and labels csv files into one file."""
    test_csv_file = Path(test_csv_file)
    test_set = pd.read_csv(test_csv_file)
    data_labels = pd.read_csv(str(test_csv_file)[:-4] + "_labels.csv")
    for category in data_labels.columns[1:]:
        test_set[category] = data_labels[category]
    if "content" in test_set.columns:
        test_set.rename(columns={"content": "comment_text"}, inplace=True)
    output_file = test_csv_file.parent / f"{test_csv_file.stem}_updated.csv"
    test_set.to_csv(output_file)
    logger.info("Updated test set saved to %s", output_file)
    return test_set


def create_val_set(csv_file, val_fraction):
    """Takes in a csv file path and creates a validation set
    out of it specified by val_fraction.
    """
    csv_file = Path(csv_file)
    dataset: pd.DataFrame = pd.read_csv(csv_file)
    np.random.seed(0)
    dataset_mod = dataset[dataset.toxic != -1]
    indices = np.random.rand(len(dataset_mod)) > val_fraction
    val_set = dataset_mod[~indices]
    output_file = csv_file.parent / "val.csv"
    logger.info("Validation set saved to %s", output_file)
    assert isinstance(val_set, (pd.DataFrame, pd.Series))
    val_set.to_csv(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str)
    parser.add_argument("--val_csv", type=str)
    parser.add_argument(
        "--update_test",
        action="store_true",
    )
    parser.add_argument(
        "--create_val_set",
        action="store_true",
    )
    args = parser.parse_args()
    if args.update_test:
        test_set = update_test(args.test_csv)
    if args.create_val_set:
        create_val_set(args.val_csv, val_fraction=0.1)
