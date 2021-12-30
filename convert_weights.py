import argparse
import hashlib
from collections import OrderedDict

import torch


def main():
    """Converts saved checkpoint to the expected format for detoxify."""
    checkpoint = torch.load(ARGS.checkpoint, map_location=ARGS.device)

    new_state_dict = {
        "state_dict": OrderedDict(),
        "config": checkpoint["hyper_parameters"]["config"],
    }
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("model."):
            k = k[6:]  # remove `model.`
        new_state_dict["state_dict"][k] = v

    torch.save(new_state_dict, ARGS.save_to)

    if ARGS.hash:
        with open(ARGS.save_to, "rb") as f:
            bytes = f.read()  # read entire file as bytes
            readable_hash = hashlib.sha256(bytes).hexdigest()
            print("Hash: ", readable_hash)

        torch.save(new_state_dict, ARGS.save_to[:-5] + f"-{readable_hash[:8]}.ckpt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="path to model checkpoint",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        help="path to save the model to",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device to load the checkpoint on",
    )
    parser.add_argument(
        "--hash",
        type=bool,
        default=True,
        help="option to save hash in name",
    )
    ARGS = parser.parse_args()
    main()
