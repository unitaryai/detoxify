import numpy as np
import pandas as pd
import json
import argparse

from sklearn.metrics import roc_auc_score


def compute_auc(y_true, y_pred):
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan


def compute_subgroup_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup]]
    return compute_auc(subgroup_examples[label], subgroup_examples[model_name])


def convert_dataframe_to_bool(df):
    bool_df = df.copy()
    for la in LANGS:
        bool_df[la] = df.lang == la
    return bool_df


def compute_lang_metrics_for_model(
    dataset, subgroups, model, label_col
):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        record = {
            "subgroup": subgroup,
            "subgroup_size": len(dataset[dataset[subgroup]]),
        }
        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)
        records.append(record)
    return pd.DataFrame(records).sort_values("subgroup_auc", ascending=True)


def main():
    with open(TEST, "r") as f:
        results = json.load(f)

    test_private_path = (
        "jigsaw_data/jigsaw-multilingual-toxic-comment-classification/updated_test.csv"
    )
    test_private = pd.read_csv(test_private_path)
    test_private = convert_dataframe_to_bool(test_private)

    test_private[MODEL_NAME] = [s[0] for s in results["scores"]]

    lang_metrics_df = compute_lang_metrics_for_model(
        test_private, LANGS, MODEL_NAME, TOXICITY_COLUMN
    )
    print(lang_metrics_df)
    lang_metrics_df.to_csv(TEST[:-9] + "_lang_breakdown.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--res_path",
        type=str,
        help="path to the saved results file",
    )
    args = parser.parse_args()
    TEST = args.res_path

    LANGS = ["es", "fr", "pt", "ru", "it", "tr"]

    TOXICITY_COLUMN = "toxic"
    MODEL_NAME = "XLM-R"
    SUBGROUP_AUC = "subgroup_auc"

    main()
