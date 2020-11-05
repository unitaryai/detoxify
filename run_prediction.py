import torch
import transformers
import pandas as pd
import argparse
from train import ToxicClassifier
import os
from collections import OrderedDict
import json
import warnings
from src.utils import get_model_and_tokenizer


def load_model_and_tokenizer(model_name, device, from_ckpt):
    if model_name is not None:
        loaded = torch.hub.load("laurahanu/detoxify", model_name)
    elif from_ckpt is not None:
        loaded = torch.load(from_ckpt, map_location=device)
    else:
        raise ValueError("A model name or a checkpoint path must be provided as input.")
    class_names = loaded["config"]["dataset"]["args"]["classes"]
    model, tokenizer = get_model_and_tokenizer(**loaded["config"]["arch"]["args"])
    model.load_state_dict(loaded["state_dict"])
    model.to(device)
    model.eval()
    return model, tokenizer, class_names


@torch.no_grad()
def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(
        model.device
    )
    out = model(**inputs)[0]
    scores = torch.sigmoid(out).cpu().detach().numpy()
    return scores


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


def run(model_name, input_obj, dest_file, device, from_ckpt):
    """Loads model from checkpoint and runs inference on the input_obj.
    Displays results as a pandas DataFrame object.
    If a dest_file is given, it saves the results to a csv file.
    """
    # parse input
    text = load_input_text(input_obj)

    # predict
    model, tokenizer, class_names = load_model_and_tokenizer(
        model_name, device, from_ckpt
    )
    scores = predict(text, model, tokenizer)

    # display results
    res = {}
    if not isinstance(text, str):
        for i, cla in enumerate(class_names):
            res[cla] = {text[ex_i]: scores[ex_i][i] for ex_i in range(len(scores))}
    else:
        res[text] = {cla: scores[0][i] for i, cla in enumerate(class_names)}

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
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run inference on (default: cuda)",
    )

    ARGS = parser.parse_args()

    assert ARGS.from_ckpt_path is not None or ARGS.model_name is not None

    if ARGS.model_name is not None:
        assert ARGS.model_name in [
            "toxic_bert",
            "toxic_roberta_bias",
            "toxic_xlmr_multilingual",
        ]

    if ARGS.from_ckpt_path is not None and ARGS.model_name is not None:
        raise ValueError(
            "Please specify only one model source, can either load model from checkpoint path or from model_name."
        )
    if ARGS.from_ckpt_path is not None:
        assert os.path.isfile(ARGS.from_ckpt_path)

    run(ARGS.model_name, ARGS.input, ARGS.save_to, ARGS.device, ARGS.from_ckpt_path)
