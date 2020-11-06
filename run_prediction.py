import torch
import transformers
import pandas as pd
import argparse
from train import ToxicClassifier
import os
from collections import OrderedDict
import json
import warnings
from detoxify import Detoxify


def load_input_text(input_obj):
    """Checks input_obj is either the path to a csv file or a text string.
    If input_obj is a csv file it returns a list of strings."""

    if isinstance(input_obj, str) and os.path.isfile(input_obj):
        if not input_obj.endswith(".csv"):
            raise ValueError("Invalid file type: only csv files supported.")
        test_set = pd.read_csv(input_obj, header=None)
        text = [t[0] for t in test_set.values.tolist()]
    elif isinstance(input_obj, str):
        text = input_obj
    else:
        raise ValueError(
            "Invalid input type: input type must be a string or a csv file."
        )
    return text


def run(model_name, input_obj, dest_file, from_ckpt):
    """Loads model from checkpoint or from model name and runs inference on the input_obj.
    Displays results as a pandas DataFrame object.
    If a dest_file is given, it saves the results to a csv file.
    """
    text = load_input_text(input_obj)
    if model_name is not None:
        res = Detoxify(model_name).predict(text)
    else:
        res = Detoxify(checkpoint=from_ckpt).predict(text)

    res_df = pd.DataFrame(res).round(3)
    print(res_df)
    if dest_file is not None:
        res_df.to_csv(dest_file)

    return res


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="text, list of strings, or csv file",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help="Name of the torch.hub model (default: None)",
    )
    parser.add_argument(
        "--from_ckpt_path",
        default=None,
        type=str,
        help="Option to load from the checkpoint path (default: False)",
    )
    parser.add_argument(
        "--save_to",
        default=None,
        type=str,
        help="destination path to output model results to (default: None)",
    )

    ARGS = parser.parse_args()

    assert ARGS.from_ckpt_path is not None or ARGS.model_name is not None

    if ARGS.model_name is not None:
        assert ARGS.model_name in [
            "original",
            "bias",
            "multilingual",
        ]

    if ARGS.from_ckpt_path is not None and ARGS.model_name is not None:
        raise ValueError(
            "Please specify only one model source, can either load model from checkpoint path or from model_name."
        )
    if ARGS.from_ckpt_path is not None:
        assert os.path.isfile(ARGS.from_ckpt_path)

    run(ARGS.model_name, ARGS.input, ARGS.save_to, ARGS.from_ckpt_path)
