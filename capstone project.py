import os
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


def load_file(candidates):
    for f in candidates:
        if os.path.exists(f):
            print(f"[OK] Loading: {f}")
            return f
    raise FileNotFoundError("Could not find:\n" + "\n".join(candidates))


# -----------------------------
# 1) Load
# -----------------------------
gdsc_path = load_file(["GDSC2-dataset.csv.zip", "GDSC_DATASET.csv.zip"])
comp_path = load_file(["Compounds-annotation.csv"])

gdsc = pd.read_csv(gdsc_path)
compounds = pd.read_csv(comp_path)

# Normalize column names
gdsc = gdsc.rename(columns={"PUTATIVE_TARGET": "TARGET", "PATHWAY_NAME": "TARGET_PATHWAY"})
compounds = compounds.rename(columns={"PUTATIVE_TARGET": "TARGET", "PATHWAY_NAME": "TARGET_PATHWAY"})

# Basic checks
for c in ["COSMIC_ID", "DRUG_ID", "TCGA_DESC", "LN_IC50"]:
    if c not in gdsc.columns:
        raise KeyError(f"Missing required GDSC column: {c}")

gdsc["DRUG_ID"] = gdsc["DRUG_ID"].astype("Int64").astype("string")
compounds["DRUG_ID"] = compounds["DRUG_ID"].astype("Int64").astype("string")

# Merge drug annotation
comp_small = compounds[[c for c in ["DRUG_ID","TARGET","TARGET_PATHWAY"] if c in compounds.columns]].drop_duplicates("DRUG_ID")
df = gdsc.merge(comp_small, on="DRUG_ID", how="left", suffixes=("", "_COMP"))

# Prefer GDSC target/pathway, else compounds
df["TARGET_FINAL"] = df["TARGET"].fillna(df.get("TARGET_COMP")).fillna("Unknown")
df["TARGET_PATHWAY_FINAL"] = df["TARGET_PATHWAY"].fillna(df.get("TARGET_PATHWAY_COMP")).fillna("Unknown")

# Target
TARGET_COL = "LN_IC50"
df = df.dropna(subset=[TARGET_COL]).copy()

# -----------------------------
# 2) FAST feature set (avoid huge cardinality)
# -----------------------------
# NOTE: DRUG_NAME is dropped to avoid huge one-hot explosion
feature_cols = [
    "TCGA_DESC",
    "DRUG_ID",
    "TARGET_FINAL",
    "TARGET_PATHWAY_FINAL",
]

feature_cols = [c for c in feature_cols if c in df.columns]
print("\n[OK] Features:", feature_cols)

# -----------------------------
# 3) Grouped split by COSMIC_ID (no leakage)
# -----------------------------
X = df[feature_cols].copy()
y = df[TARGET_COL].astype(float).to_numpy()
groups = df["COSMIC_ID"].astype(str).to_numpy()

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# -----------------------------
# 4) Speed trick: sample training rows for iteration
# -----------------------------
MAX_TRAIN_ROWS = 80000  # change to 120000 if your laptop can handle it

if len(X_train) > MAX_TRAIN_ROWS:
    samp = np.random.RandomState(42).choice(len(X_train), size=MAX_TRAIN_ROWS, replace=False)
    X_train = X_train.iloc[samp]
    y_train = y_train[samp]
    print(f"[FAST MODE] Sampled training to {MAX_TRAIN_ROWS} rows")

print(f"[INFO] Train={len(X_train)} Test={len(X_test)}")

# -----------------------------
# 5) Preprocess (reduce rare categories)
# -----------------------------
categorical_cols = feature_cols

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(
    handle_unknown="ignore",
    min_frequency=50,
    sparse_output=False   # ← ADD THIS
)
            )
        ]), categorical_cols)
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

# -----------------------------
# 6) FAST models
# -----------------------------
models = {
    "Ridge (fast baseline)": Ridge(alpha=2.0, random_state=42),
    "HistGradientBoosting (strong)": HistGradientBoostingRegressor(
        learning_rate=0.07,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42
    )
}

# -----------------------------
# 7) Train + Evaluate
# -----------------------------
for name, model in models.items():
    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))

    print(f"\n===== {name} =====")
    print("RMSE:", rmse)
    print("MAE :", mae)
    print("R2  :", r2)

  #  sample = X_test.copy()
   # sample["Actual_LN_IC50"] = np.asarray(y_test)
    #sample["Predicted_LN_IC50"] = model.predict(X_test)
   # sample["Interpretation"] = sample["Predicted_LN_IC50"].apply(
    #    lambda x: "High sensitivity" if x < 0 else "Low sensitivity"
    #)
    #print(sample[["Actual_LN_IC50","Predicted_LN_IC50","Interpretation"]].head(10))
  # Create a small sample table for inspection
n_show = 25
sample = X_test.head(n_show).copy()

sample["Actual_LN_IC50"] = y_test[:n_show]
sample["Pred_LN_IC50"] = pred[:n_show]

print("\n=== Sample predictions (first 25 rows) ===")
print(sample[["TCGA_DESC", "DRUG_ID", "TARGET_FINAL", "TARGET_PATHWAY_FINAL", "Actual_LN_IC50", "Pred_LN_IC50"]])  
