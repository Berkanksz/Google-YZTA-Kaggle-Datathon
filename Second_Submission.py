import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET = "bilissel_performans_skoru"
ID_COL = "id"
N_SPLITS = 5
RANDOM_STATE = 42
OUTPUT_PATH = "submission.csv"


def find_file(filename):
    """Find files both on Kaggle and in the local workspace."""
    roots = ["/kaggle/input", ".", "/Users/berkanoksuz/Desktop/KaggleGoogle Codex"]
    for root in roots:
        if not os.path.exists(root):
            continue
        for dirname, _, files in os.walk(root):
            if filename in files:
                return os.path.join(dirname, filename)
    raise FileNotFoundError(f"{filename} not found")


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, np.clip(y_pred, 0, 10)) ** 0.5


def prepare_catboost(train_df, valid_df, cat_cols):
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    for col in cat_cols:
        train_df[col] = train_df[col].fillna("__MISSING__").astype(str)
        valid_df[col] = valid_df[col].fillna("__MISSING__").astype(str)
    return train_df, valid_df


def prepare_lightgbm(train_df, valid_df, cat_cols):
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    for col in cat_cols:
        categories = (
            pd.concat([train_df[col], valid_df[col]], ignore_index=True)
            .fillna("__MISSING__")
            .astype(str)
            .unique()
            .tolist()
        )
        train_df[col] = pd.Categorical(
            train_df[col].fillna("__MISSING__").astype(str), categories=categories
        )
        valid_df[col] = pd.Categorical(
            valid_df[col].fillna("__MISSING__").astype(str), categories=categories
        )
    return train_df, valid_df


def fit_catboost_model(name, params, X, y, X_test, cat_cols, folds):
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))

    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        X_train, X_valid = prepare_catboost(
            X.iloc[train_idx], X.iloc[valid_idx], cat_cols
        )
        X_test_fold, _ = prepare_catboost(X_test, X_test, cat_cols)

        model = CatBoostRegressor(**params)
        model.fit(
            X_train,
            y[train_idx],
            cat_features=cat_cols,
            eval_set=(X_valid, y[valid_idx]),
            verbose=False,
        )

        oof[valid_idx] = model.predict(X_valid)
        test_pred += model.predict(X_test_fold) / len(folds)

        fold_rmse = rmse(y[valid_idx], oof[valid_idx])
        print(
            f"{name} fold {fold}: RMSE={fold_rmse:.6f}, "
            f"best_iter={model.get_best_iteration()}",
            flush=True,
        )

    print(f"{name} OOF RMSE: {rmse(y, oof):.6f}", flush=True)
    return oof, test_pred


def fit_lightgbm_model(name, params, X, y, X_test, cat_cols, folds):
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))

    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        X_train, X_valid = prepare_lightgbm(
            X.iloc[train_idx], X.iloc[valid_idx], cat_cols
        )
        _, X_test_fold = prepare_lightgbm(X.iloc[train_idx], X_test, cat_cols)

        model = LGBMRegressor(**params)
        model.fit(
            X_train,
            y[train_idx],
            categorical_feature=cat_cols,
            eval_set=[(X_valid, y[valid_idx])],
            eval_metric="rmse",
        )

        oof[valid_idx] = model.predict(X_valid)
        test_pred += model.predict(X_test_fold) / len(folds)

        fold_rmse = rmse(y[valid_idx], oof[valid_idx])
        print(f"{name} fold {fold}: RMSE={fold_rmse:.6f}", flush=True)

    print(f"{name} OOF RMSE: {rmse(y, oof):.6f}", flush=True)
    return oof, test_pred


def fit_ridge_model(name, X, y, X_test, cat_cols, num_cols, folds):
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="__MISSING__"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                cat_cols,
            ),
        ]
    )

    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        model = make_pipeline(
            preprocessor,
            RidgeCV(alphas=np.logspace(-3, 4, 20)),
        )
        model.fit(X.iloc[train_idx], y[train_idx])

        oof[valid_idx] = model.predict(X.iloc[valid_idx])
        test_pred += model.predict(X_test) / len(folds)

        fold_rmse = rmse(y[valid_idx], oof[valid_idx])
        print(f"{name} fold {fold}: RMSE={fold_rmse:.6f}", flush=True)

    print(f"{name} OOF RMSE: {rmse(y, oof):.6f}", flush=True)
    return oof, test_pred


def learn_blend_weights(model_names, oof_predictions, y):
    clipped_oof = np.column_stack([np.clip(oof_predictions[name], 0, 10) for name in model_names])
    blender = LinearRegression(positive=True, fit_intercept=False)
    blender.fit(clipped_oof, y)

    weights = blender.coef_.astype(float)
    if weights.sum() <= 0:
        weights = np.ones(len(model_names), dtype=float)
    weights = weights / weights.sum()

    blend_oof = clipped_oof @ weights
    print("\nBlend weights:", flush=True)
    for name, weight in zip(model_names, weights):
        print(f"  {name}: {weight:.4f}", flush=True)
    print(f"Blended OOF RMSE: {rmse(y, blend_oof):.6f}", flush=True)

    return dict(zip(model_names, weights))


def main():
    train_path = find_file("train.csv")
    test_path = find_file("test_x.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X = train.drop(columns=[TARGET, ID_COL])
    y = train[TARGET].to_numpy()
    X_test = test.drop(columns=[ID_COL])

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    folds = list(KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(X))

    cat_common = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "od_type": "Iter",
        "od_wait": 100,
        "allow_writing_files": False,
        "thread_count": -1,
    }

    model_outputs = {}
    test_outputs = {}

    model_outputs["cat_d6"], test_outputs["cat_d6"] = fit_catboost_model(
        "cat_d6",
        {
            **cat_common,
            "iterations": 2500,
            "learning_rate": 0.035,
            "depth": 6,
            "l2_leaf_reg": 3,
            "random_seed": 42,
        },
        X,
        y,
        X_test,
        cat_cols,
        folds,
    )

    model_outputs["cat_d8"], test_outputs["cat_d8"] = fit_catboost_model(
        "cat_d8",
        {
            **cat_common,
            "iterations": 1800,
            "learning_rate": 0.035,
            "depth": 8,
            "l2_leaf_reg": 4,
            "random_seed": 123,
        },
        X,
        y,
        X_test,
        cat_cols,
        folds,
    )

    model_outputs["lgb600"], test_outputs["lgb600"] = fit_lightgbm_model(
        "lgb600",
        {
            "n_estimators": 600,
            "learning_rate": 0.035,
            "num_leaves": 31,
            "min_child_samples": 30,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.05,
            "reg_lambda": 0.1,
            "random_state": 77,
            "n_jobs": -1,
            "verbosity": -1,
        },
        X,
        y,
        X_test,
        cat_cols,
        folds,
    )

    model_outputs["ridge"], test_outputs["ridge"] = fit_ridge_model(
        "ridge", X, y, X_test, cat_cols, num_cols, folds
    )

    model_names = list(model_outputs)
    weights = learn_blend_weights(model_names, model_outputs, y)

    final_pred = np.zeros(len(X_test))
    for name in model_names:
        final_pred += weights[name] * np.clip(test_outputs[name], 0, 10)
    final_pred = np.clip(final_pred, 0, 10)

    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: final_pred})
    submission.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved {OUTPUT_PATH}: {submission.shape}", flush=True)
    print(submission.head(), flush=True)


if __name__ == "__main__":
    main()
