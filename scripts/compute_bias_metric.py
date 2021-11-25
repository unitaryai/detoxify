import pandas as pd
import json
import argparse

from detoxify.bias_metrics import convert_dataframe_to_bool, MODEL_NAME, compute_bias_metrics_for_model, \
    IDENTITY_COLUMNS, TOXICITY_COLUMN, get_final_metric, calculate_overall_auc


def main():
    with open(TEST, "r") as f:
        results = json.load(f)

    test_private_path = "jigsaw_data/jigsaw-unintended-bias-in-toxicity-classification/test_private_expanded.csv"
    test_private = pd.read_csv(test_private_path)
    test_private = convert_dataframe_to_bool(test_private)

    test_private[MODEL_NAME] = [s[0] for s in results["scores"]]

    bias_metrics_df = compute_bias_metrics_for_model(
        test_private, IDENTITY_COLUMNS, MODEL_NAME, TOXICITY_COLUMN
    )
    print(bias_metrics_df)

    final_metric = get_final_metric(
        bias_metrics_df, calculate_overall_auc(test_private, MODEL_NAME)
    )
    print(final_metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "res_path",
        type=str,
        help="path to the saved results file",
    )
    args = parser.parse_args()

    TEST = args.res_path

    main()
