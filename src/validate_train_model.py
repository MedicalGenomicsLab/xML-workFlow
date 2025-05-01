# --- 100 characters ------------------------------------------------------------------------------
# Created by: Khoa Tran | 2025-04-23 | Validate trained model by re-training on the same data
# --------------------------------------------------------------------------------------------------
# -*- coding: utf-8 -*-

import argparse

import scanpy as sc
import pickle as pkl

from pathlib import Path
from utils.file_io import check_file_path
from utils.constant import UNIVERSAL_RANDOM_STATE, TARGET_COL


def main(
    search_obj_path: str | Path,
    ml_model: str,
    train_data_path: str | Path,
) -> None:
    # Read trained search object and get best estimator + features
    search_obj = pkl.load(search_obj_path)
    clf = search_obj.best_estimator_
    feats = clf.feature_names_in_
    print(f"Validate trained model: {ml_model} with features:\n{feats}")

    # Load train data without using prepare_train_data() function
    train_adata = sc.read(train_data_path)
    train_df = train_adata.to_df()

    # Manually shuffle the training data
    X_train = train_df.sample(frac=1, axis=0, random_state=UNIVERSAL_RANDOM_STATE)[
        feats
    ]
    y_train = train_df.loc[X_train.index, TARGET_COL]

    # First get predictions on the training data
    y_train_pred_1 = clf.predict(X_train)
    y_train_proba_1 = clf.predict_proba(X_train)

    # Then retrain the model on the training data
    clf.fit(X_train, y_train)
    y_train_pred_2 = clf.predict(X_train)
    y_train_proba_2 = clf.predict_proba(X_train)

    # If the model was trained appropriately, the predictions should be the same
    try:
        assert (y_train_pred_1 == y_train_pred_2).all() and (
            y_train_proba_1 == y_train_proba_2
        ).all()
    except AssertionError as e:
        print(f"Model was not trained appropriately: {e}")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make predictions on test data using trained sklearn models"
    )
    parser.add_argument(
        "--search_obj_path",
        type=lambda x: check_file_path(x, file_type=[".pkl", ".pickle"]),
        required=True,
        help="Path to the search object",
    )
    parser.add_argument(
        "--ml_model",
        type=str,
        required=True,
        help="Name of model to train. Must exist in src.utils.params.clfs",
    )
    parser.add_argument(
        "--train_data_path",
        type=lambda x: check_file_path(x, file_type=".h5ad"),
        required=True,
        help="Path to the training data",
    )
    args = parser.parse_args()

    main(
        search_obj_path=args.search_obj_path,
        ml_model=args.ml_model,
        train_data_path=args.train_data_path,
    )
