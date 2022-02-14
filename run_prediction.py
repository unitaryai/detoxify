import argparse
import os

import pandas as pd
from detoxify import Detoxify


def load_input_text(input_obj):
    """Checks input_obj is either the path to a txt file or a text string.
    If input_obj is a txt file it returns a list of strings."""

    if isinstance(input_obj, str) and os.path.isfile(input_obj):
        if not input_obj.endswith(".txt"):
            raise ValueError("Invalid file type: only txt files supported.")
        text = open(input_obj).read().splitlines()
    elif isinstance(input_obj, str):
        text = input_obj
    else:
        raise ValueError("Invalid input type: input type must be a string or a txt file.")
    return text


def run(model_name, input_obj, dest_file, from_ckpt, device="cpu"):
    """Loads model from checkpoint or from model name and runs inference on the input_obj.
    Displays results as a pandas DataFrame object.
    If a dest_file is given, it saves the results to a txt file.
    """
    text = load_input_text(input_obj)
    if model_name is not None:
        model = Detoxify(model_name, device=device)
    else:
        model = Detoxify(checkpoint=from_ckpt, device=device)
    res = model.predict(text)

    res_df = pd.DataFrame(res, index=[text] if isinstance(text, str) else text).round(5)
    print(res_df)
    if dest_file is not None:
        res_df.index.name = "input_text"
        res_df.to_csv(dest_file)

    return res


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="text, list of strings, or txt file",
    )
    parser.add_argument(
        "--model_name",
        default="unbiased",
        type=str,
        help="Name of the torch.hub model (default: unbiased)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to load the model on",
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

    args = parser.parse_args()

    assert args.from_ckpt_path is not None or args.model_name is not None

    if args.model_name is not None:
        assert args.model_name in [
            "original",
            "unbiased",
            "multilingual",
        ]

    if args.from_ckpt_path is not None and args.model_name is not None:
        raise ValueError(
            "Please specify only one model source, can either load model from checkpoint path or from model_name."
        )
    if args.from_ckpt_path is not None:
        assert os.path.isfile(args.from_ckpt_path)

    run(
        args.model_name,
        args.input,
        args.save_to,
        args.from_ckpt_path,
        device=args.device,
    )
