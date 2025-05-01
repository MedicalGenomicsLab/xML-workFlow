# --- 100 characters ------------------------------------------------------------------------------
# Created by: Khoa Tran | 2025-04-23 | Make predictions on test data using trained sklearn models
# --------------------------------------------------------------------------------------------------
# -*- coding: utf-8 -*-

import mlflow
import argparse

import pandas as pd
import pickle as pkl

from typing import Tuple, List
from pathlib import Path
from sklearn.metrics import roc_curve

from utils.params import clfs_df
from utils.file_io import check_file_path
from utils.constant import PREDICT_LABELS
from utils.pred_func import calculate_prediction_metrics, prepare_test_data


def main(
    search_obj_path: str | Path,
    ml_model: str,
    test_data_path: str | Path,
    output_dir: str | Path,
    save_output: bool = True,
) -> Tuple[
    list,
    list,
    list,
    list,
]:
    """Main function to run predictions on test dataset using the best model from gridsearchcv

    Args:
        - search_obj_path (str|Path):       Path to the search object
        - model (str):                      Name of model to train. Must exist in src.utils.params.clfs
        - test_data_path (str|Path):        Path to the test data

    Returns:
        - None

    Example:
        >>> python src/pred.py \
        >>> --search_obj-path path/to/search_obj.pkl \
        >>> --ml-model random_forest \
        >>> --test-data-path path/to/test_data.h5ad \
        >>> --output-dir path/to/output_dir \
        >>> --validate-trained-model True
    """
    mlflow.autolog()

    # Read trained search object and get best estimator + features
    search_obj = pkl.load(open(search_obj_path, "rb"))
    clf = search_obj.best_estimator_
    feats = clf.feature_names_in_
    print(f"Running predictions for {ml_model} with features:\n{feats}")

    # Get test data
    X_test, y_test = prepare_test_data(
        test_data_path=test_data_path,
        feats=feats,
    )

    # Get prediction labels and probabilities
    y_test_pred, y_test_proba = clf.predict(X_test), clf.predict_proba(X_test)

    # Colect predictions vs groundtruth vs prediction probabilities
    pred_proba_df = pd.DataFrame(
        {
            "model": ml_model,
            "y_truth": y_test,
            "y_pred": y_test_pred,
            "y_pred_proba": y_test_proba[:, 0],
        }
    )

    # Get ROC curve table (FPR vs TPR at different prediction thresholds)
    fpr, tpr, thresholds = roc_curve(
        y_true=y_test,
        y_score=y_test_proba[:, 0],
        pos_label=PREDICT_LABELS[0],
    )
    roc_curve_df = pd.DataFrame(
        {
            "model": ml_model,
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
        }
    )

    # Collect prediction metrics
    pred_metrics_ser = calculate_prediction_metrics(
        y_true=y_test.to_numpy(),
        y_pred=y_test_pred,
        y_pred_proba=y_test_proba,
        global_average_method="weighted",
        ml_model=ml_model,
    )

    # Collect predictions in pandas Series
    y_test_ser = pd.Series(y_test_pred, index=X_test.index)

    # Save predictions to output directory
    if save_output:
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_proba_df.to_csv(
            output_dir.joinpath(f"{ml_model}_pred_proba").with_suffix(".tsv"), sep="\t"
        )
        roc_curve_df.to_csv(
            output_dir.joinpath(f"{ml_model}_roc_curve").with_suffix(".tsv"), sep="\t"
        )
        pred_metrics_ser.to_csv(
            output_dir.joinpath(f"{ml_model}_pred_metrics").with_suffix(".tsv"),
            sep="\t",
        )
        y_test_ser.to_csv(
            output_dir.joinpath(f"{ml_model}_pred").with_suffix(".tsv"), sep="\t"
        )

    # Return predictions
    return y_test_ser, pred_proba_df, pred_metrics_ser, roc_curve_df


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description=__doc__)

    # Required arguments
    parser.add_argument(
        "--search-obj-path",
        # Path to search object must be a pickle file
        type=lambda x: check_file_path(x, [".pkl", ".pickle"]),
        required=True,
        help="Path to search object",
    )
    parser.add_argument(
        "--ml_model",
        type=str,
        required=True,
        choices=clfs_df["model"].tolist(),
        help="Name of ML model. Must exist in utils.params.clfs",
    )
    parser.add_argument(
        "--output-dir",
        type=str | Path,
        required=True,
        help="Path to output directory",
    )

    # Optional arguments
    parser.add_argument(
        "--train-data-path",
        type=lambda x: check_file_path(x, [".csv", ".tsv", ".h5ad"]),
        required=True,
        help="Path to features metadata",
    )
    parser.add_argument(
        "--test-data-path",
        type=lambda x: check_file_path(x, [".csv", ".tsv", ".h5ad"]),
        required=True,
        help="Path to features metadata",
    )

    # Parse arguments
    args, unknowns = parser.parse_known_args()

    # Run main function
    main(
        search_obj_path=args.search_obj_path,
        model=args.model,
    )
