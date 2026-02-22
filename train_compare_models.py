"""
Train & compare multiple regression models for drug sensitivity.

Models included:
- Ridge (linear baseline on one-hot)
- RandomForestRegressor (nonlinear baseline)
- HistGradientBoostingRegressor (strong tabular baseline)

Evaluation:
- Grouped split by cell line (to reduce leakage) + grouped CV
- Metrics: RMSE / MAE / R2

Outputs:
- artifacts/leaderboard.csv
- artifacts/best_model.joblib  (best by RMSE)
- artifacts/metrics.json
- artifacts/feature_info.json

USAGE
-----
python train_compare_models.py \
  --model_table "/mnt/data/gdsc_ds_project/data/model_table.parquet" \
  --target_col "LN_IC50" \
  --out_dir "/mnt/data/gdsc_ds_project/artifacts"

Tip:
Run build_dataset.py first to create the parquet model table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import joblib


def rmse(y_true, y_pred) -> float:
    return float(mean_squared_error(y_true, y_pred, squared=False))


def build_preprocessor(categorical_cols: list[str], numeric_cols: list[str]):
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
    ])
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, categorical_cols),
            ("num", num_pipe, numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def evaluate_model(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_splits: int = 5):
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups), start=1):
        pipe.fit(X.iloc[tr], y[tr])
        pred = pipe.predict(X.iloc[va])
        rows.append({
            "fold": fold,
            "rmse": rmse(y[va], pred),
            "mae": float(mean_absolute_error(y[va], pred)),
            "r2": float(r2_score(y[va], pred)),
            "n_val": int(len(va)),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_table", type=str, required=True)
    ap.add_argument("--target_col", type=str, default="LN_IC50", choices=["LN_IC50", "AUC"])
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.model_table)

    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in model table.")

    df = df.dropna(subset=[args.target_col]).copy()

    # Choose a leakage-resistant grouping key
    group_col = "COSMIC_ID" if "COSMIC_ID" in df.columns else ("CELL_LINE_NAME" if "CELL_LINE_NAME" in df.columns else None)
    if group_col is None:
        raise ValueError("No grouping column found (COSMIC_ID or CELL_LINE_NAME).")

    groups = df[group_col].astype("string").to_numpy()

    # Feature set: include interpretable clinical proxies + drug annotations
    base_features = [
        "TCGA_DESC",
        "TISSUE_DESC_1",
        "TISSUE_DESC_2",
        "CANCER_TYPE_TCGA_MATCH",
        "MSI_STATUS",
        "GROWTH_PROPERTIES",
        "SCREEN_MEDIUM",
        "TARGET",
        "TARGET_PATHWAY",
        "DRUG_ID",
        "DRUG_NAME",
        # Optional extra drug annotation columns if present (kept categorical)
        "TARGET_PATHWAY_ANNOT",
    ]
    feature_cols = [c for c in base_features if c in df.columns]

    # Build X/y
    X = df[feature_cols].copy()
    y = df[args.target_col].astype(float).to_numpy()

    # Enforce categorical types
    for c in feature_cols:
        if c == "DRUG_ID":
            X[c] = X[c].astype("string")
        elif X[c].dtype != "float64" and X[c].dtype != "int64":
            X[c] = X[c].astype("string")

    # Identify numeric columns (if any sneak in)
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    pre = build_preprocessor(categorical_cols, numeric_cols)

    # Candidate models
    candidates = [
        ("Ridge", Ridge(alpha=2.0, random_state=args.random_state)),
        ("RandomForest", RandomForestRegressor(
            n_estimators=400, random_state=args.random_state, n_jobs=-1,
            max_depth=None, min_samples_leaf=10
        )),
        ("HistGB", HistGradientBoostingRegressor(
            learning_rate=0.07, max_leaf_nodes=31, min_samples_leaf=20,
            random_state=args.random_state
        )),
    ]

    leaderboard = []
    per_model_cv = {}

    # CV evaluation (grouped)
    for name, model in candidates:
        pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])
        cv = evaluate_model(pipe, X, y, groups, n_splits=5)
        per_model_cv[name] = cv
        leaderboard.append({
            "model": name,
            "cv_rmse_mean": float(cv["rmse"].mean()),
            "cv_rmse_std": float(cv["rmse"].std(ddof=1)),
            "cv_mae_mean": float(cv["mae"].mean()),
            "cv_r2_mean": float(cv["r2"].mean()),
        })

    lb = pd.DataFrame(leaderboard).sort_values("cv_rmse_mean", ascending=True)
    lb.to_csv(out_dir / "leaderboard.csv", index=False)

    # Train/test evaluation (grouped split)
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    tr_idx, te_idx = next(splitter.split(X, y, groups=groups))
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    best_name = lb.iloc[0]["model"]
    best_model = dict(candidates)[best_name]
    best_pipe = Pipeline(steps=[("preprocess", pre), ("model", best_model)])
    best_pipe.fit(X_tr, y_tr)
    pred = best_pipe.predict(X_te)

    test_metrics = {
        "target": args.target_col,
        "group_col": group_col,
        "n_rows": int(len(df)),
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "test_rmse": rmse(y_te, pred),
        "test_mae": float(mean_absolute_error(y_te, pred)),
        "test_r2": float(r2_score(y_te, pred)),
        "best_model": best_name,
        "note": "Research model on cell-line screening data; not clinically validated.",
    }

    (out_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2))
    (out_dir / "feature_info.json").write_text(json.dumps({
        "feature_cols": feature_cols,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
    }, indent=2))

    joblib.dump(best_pipe, out_dir / "best_model.joblib")

    # Save CV fold tables (nice for report)
    for name, cv in per_model_cv.items():
        cv.to_csv(out_dir / f"cv_{name}.csv", index=False)

    print("Saved leaderboard:", out_dir / "leaderboard.csv")
    print("Saved best model:", out_dir / "best_model.joblib")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
