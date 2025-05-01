# --- 100 characters ------------------------------------------------------------------------------
# Created by: Khoa Tran | 2025-04-23 | Make predictions on test data using trained sklearn models
# --------------------------------------------------------------------------------------------------
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scanpy as sc

from pathlib import Path
from typing import Tuple, List
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Utilities
from .constant import (
    TARGET_COL,
    PREDICT_LABELS,
)


def prepare_test_data(
    test_data_path: str | Path,
    feats: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare test data for training.

    Args:
        - test_data_path (str):     Path to test data
        - feats (List[str]):        List of features to use for training

    Returns:
        - X_test (pd.DataFrame):    Test data
        - y_test (pd.DataFrame):    Test labels
    """
    # Load test data (need to be a .h5ad file)
    try:
        test_adata = sc.read(test_data_path)
        test_df = test_adata.to_df()
    except Exception as e:
        raise ValueError(f"Invalid data type: {e}")

    # X_test are columns != TARGET / UNUSED
    X_test, y_test = test_df[feats], test_df[TARGET_COL]

    return X_test, y_test


def calculate_prediction_metrics(
    y_true: np.array,
    y_pred: np.array,
    y_pred_proba: np.ndarray,
    ml_model: str,
    global_average_method: str = "weighted",
    labels: List = PREDICT_LABELS,
) -> Tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    np.array,
]:
    """Calculate prediction metrics.

    Args:
        - y_true: True labels
        - y_pred: Predicted labels
        - y_pred_proba: Predicted probabilities
    """
    # Calculate accuracy,
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)

    # Calculate global precision, recall, and F1
    global_prec = precision_score(
        y_true=y_true,
        y_pred=y_pred,
        average=global_average_method,
        labels=labels,
    )
    global_rec = recall_score(
        y_true=y_true,
        y_pred=y_pred,
        average=global_average_method,
        labels=labels,
    )
    global_f1 = f1_score(
        y_true=y_true,
        y_pred=y_pred,
        average=global_average_method,
        labels=labels,
    )

    # Calculate ROC-AUC and confusion matrix
    roc_auc = roc_auc_score(
        y_true=y_true,
        y_score=y_pred_proba[:, 1],  # roc_auc_score always takes 2nd column
        average=global_average_method,
        labels=labels,
    )
    confusion = confusion_matrix(y_pred=y_pred, y_true=y_true)

    # Calculate precision, recall, and F1 for Good response prediction
    good_prec = precision_score(
        y_true=y_true,
        y_pred=y_pred,
        average="binary",
        pos_label=PREDICT_LABELS[1],
        labels=labels,
    )
    good_rec = recall_score(
        y_true=y_true,
        y_pred=y_pred,
        average="binary",
        pos_label=PREDICT_LABELS[1],
        labels=labels,
    )
    good_f1 = f1_score(
        y_true=y_true,
        y_pred=y_pred,
        average="binary",
        pos_label=PREDICT_LABELS[1],
        labels=labels,
    )

    # Calculate precision, recall, and F1 for Poor response prediction
    poor_prec = precision_score(
        y_true=y_true,
        y_pred=y_pred,
        average="binary",
        pos_label=PREDICT_LABELS[0],
        labels=labels,
    )
    poor_rec = recall_score(
        y_true=y_true,
        y_pred=y_pred,
        average="binary",
        pos_label=PREDICT_LABELS[0],
        labels=labels,
    )
    poor_f1 = f1_score(
        y_true=y_true,
        y_pred=y_pred,
        average="binary",
        pos_label=PREDICT_LABELS[0],
        labels=labels,
    )

    return pd.Series(
        {
            "accuracy": acc,
            "roc_auc": roc_auc,
            "global_prec": global_prec,
            "global_rec": global_rec,
            "global_f1": global_f1,
            "good_prec": good_prec,
            "good_rec": good_rec,
            "good_f1": good_f1,
            "poor_prec": poor_prec,
            "poor_rec": poor_rec,
            "poor_f1": poor_f1,
            "confusion": confusion,
        },
        name=ml_model,
    )
