import os
import zipfile
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression, ElasticNet
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def load_zip_csv(path):
    with zipfile.ZipFile(path, "r") as z:
        csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        return pd.read_csv(z.open(csv_name), low_memory=False)


df1       = load_zip_csv("GDSC_DATASET.csv.zip")
df2       = load_zip_csv("GDSC2-dataset.csv.zip")
compounds = pd.read_csv("Compounds-annotation.csv")

df1 = df1.dropna(subset=["TCGA_DESC"])
df2 = df2.dropna(subset=["TCGA_DESC"])

gdsc = pd.merge(df1, df2, on=["COSMIC_ID", "DRUG_ID"], how="inner", suffixes=("", "_df2"))
print(f"Merged GDSC shape: {gdsc.shape}")

# Pick the right column names after merge
TARGET_COL = "LN_IC50"   if "LN_IC50"   in gdsc.columns else "LN_IC50_df2"
TCGA_COL   = "TCGA_DESC" if "TCGA_DESC" in gdsc.columns else "TCGA_DESC_df2"


# Merge compound annotations
compounds = compounds.rename(columns={"PUTATIVE_TARGET": "TARGET", "PATHWAY_NAME": "TARGET_PATHWAY"})
gdsc["DRUG_ID"]      = gdsc["DRUG_ID"].astype("Int64").astype("string")
compounds["DRUG_ID"] = compounds["DRUG_ID"].astype("Int64").astype("string")

comp_small = compounds[["DRUG_ID", "TARGET", "TARGET_PATHWAY"]].drop_duplicates("DRUG_ID")
gdsc = gdsc.merge(comp_small, on="DRUG_ID", how="left", suffixes=("", "_COMP"))

gdsc["TARGET_FINAL"]         = gdsc.get("TARGET",         pd.Series()).fillna(gdsc.get("TARGET_COMP",         pd.Series())).fillna("Unknown")
gdsc["TARGET_PATHWAY_FINAL"] = gdsc.get("TARGET_PATHWAY", pd.Series()).fillna(gdsc.get("TARGET_PATHWAY_COMP", pd.Series())).fillna("Unknown")

gdsc = gdsc.dropna(subset=[TARGET_COL]).copy()


# Features & grouped train/test split
feature_cols = [c for c in [TCGA_COL, "DRUG_ID", "TARGET_FINAL", "TARGET_PATHWAY_FINAL"] if c in gdsc.columns]
print(f"Features: {feature_cols}  |  Rows: {len(gdsc)}")

X      = gdsc[feature_cols].copy()
y      = gdsc[TARGET_COL].astype(float).to_numpy()
groups = gdsc["COSMIC_ID"].astype(str).to_numpy()

train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y[train_idx],      y[test_idx]

# Subsample for speed
MAX_TRAIN_ROWS = 80_000
if len(X_train) > MAX_TRAIN_ROWS:
    idx = np.random.RandomState(42).choice(len(X_train), MAX_TRAIN_ROWS, replace=False)
    X_train, y_train = X_train.iloc[idx], y_train[idx]

print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")


# Pipeline & models
preprocessor = ColumnTransformer([
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False)),
    ]), feature_cols)
])

models = {
    "Linear Regression":    LinearRegression(),
    "Ridge":                Ridge(alpha=2.0, random_state=42),
    "ElasticNet":           ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000, random_state=42),
    "HistGradientBoosting": HistGradientBoostingRegressor(
        learning_rate=0.07, max_leaf_nodes=31, min_samples_leaf=20, random_state=42
    ),
    "LightGBM":             LGBMRegressor(
        learning_rate=0.07, num_leaves=31, min_child_samples=20, random_state=42
    ),
    "CatBoost":             CatBoostRegressor(
        learning_rate=0.07, depth=6, iterations=500, random_state=42, verbose=0
    ),
}

# Train & evaluate
for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    print(f"\n {name} ")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_test, pred)):.4f}")
    print(f"MAE  : {mean_absolute_error(y_test, pred):.4f}")
    print(f"R²   : {r2_score(y_test, pred):.4f}")

# Sample predictions from the last model
sample = X_test.head(25).copy()
sample["Actual_LN_IC50"] = y_test[:25]
sample["Pred_LN_IC50"]   = pred[:25]
print("\n Sample Predictions (first 25 rows)")
print(sample.to_string(index=False))