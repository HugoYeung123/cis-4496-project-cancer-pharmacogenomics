"""
Explainability utilities for the trained sklearn pipeline.

Produces:
- permutation_importance.csv (global importance)
- ridge_coefficients.csv (if model is Ridge; top +/- coefficients)

USAGE
-----
python explain_model.py \
  --model_path "/mnt/data/gdsc_ds_project/artifacts/best_model.joblib" \
  --model_table "/mnt/data/gdsc_ds_project/data/model_table.parquet" \
  --target_col "LN_IC50" \
  --out_dir "/mnt/data/gdsc_ds_project/artifacts"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
import joblib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--model_table", type=str, required=True)
    ap.add_argument("--target_col", type=str, default="LN_IC50", choices=["LN_IC50", "AUC"])
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_repeats", type=int, default=8)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = joblib.load(args.model_path)
    df = pd.read_parquet(args.model_table).dropna(subset=[args.target_col]).copy()

    # Use the same feature list stored during training if available
    feat_path = out_dir / "feature_info.json"
    if feat_path.exists():
        feature_cols = pd.read_json(feat_path)["feature_cols"].tolist() if False else None  # keep robust
        # easier: read manually
        import json
        feature_cols = json.loads(feat_path.read_text())["feature_cols"]
    else:
        # fallback: common columns
        feature_cols = [c for c in ["TCGA_DESC","TISSUE_DESC_1","TISSUE_DESC_2","MSI_STATUS","TARGET","TARGET_PATHWAY","DRUG_ID","DRUG_NAME"] if c in df.columns]

    X = df[feature_cols].copy()
    y = df[args.target_col].astype(float).to_numpy()

    # Permutation importance on a subsample for speed
    n = min(len(df), 5000)
    sub = df.sample(n=n, random_state=args.random_state)
    Xs = sub[feature_cols].copy()
    ys = sub[args.target_col].astype(float).to_numpy()

    result = permutation_importance(
        pipe, Xs, ys,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
        scoring="neg_root_mean_squared_error",
    )

    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    imp.to_csv(out_dir / "permutation_importance.csv", index=False)
    print("Saved:", out_dir / "permutation_importance.csv")

    # Ridge coefficients (on transformed feature space)
    model = pipe.named_steps.get("model", None)
    preprocess = pipe.named_steps.get("preprocess", None)
    if model is not None and model.__class__.__name__ == "Ridge" and preprocess is not None:
        # get feature names from one-hot
        try:
            names = preprocess.get_feature_names_out()
            coef = model.coef_.ravel()
            coef_df = pd.DataFrame({"feature": names, "coef": coef})
            coef_df["abs_coef"] = coef_df["coef"].abs()
            coef_df = coef_df.sort_values("abs_coef", ascending=False).head(300)
            coef_df.to_csv(out_dir / "ridge_coefficients.csv", index=False)
            print("Saved:", out_dir / "ridge_coefficients.csv")
        except Exception as e:
            print("Could not extract ridge coefficients:", e)


if __name__ == "__main__":
    main()
