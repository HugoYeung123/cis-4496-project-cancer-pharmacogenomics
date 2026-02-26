import zipfile
import pandas as pd
import os
from sklearn.preprocessing import OrdinalEncoder

ZIP_PATH_DF  = r"C:\Users\moham\OneDrive\Desktop\dataset\GDSC_DATASET.csv.zip"
ZIP_PATH_DF2 = r"C:\Users\moham\OneDrive\Desktop\dataset\GDSC2-dataset.csv.zip"

def load_zip_csv(path):
    z = zipfile.ZipFile(path, "r")

    print("\nFiles inside", path)
    print(z.namelist())

    csv_name = None
    for name in z.namelist():
        if name.lower().endswith(".csv"):
            csv_name = name
            break


    with z.open(csv_name) as f:
        df = pd.read_csv(f, low_memory=False)

    return df


df  = load_zip_csv(ZIP_PATH_DF)
df2 = load_zip_csv(ZIP_PATH_DF2)

print("\ndf shape :", df.shape)
print("df2 shape:", df2.shape)

TARGET_COL = "LN_IC50"
TCGA_COL   = "TCGA_DESC"

print("\nclean df")
df = df.dropna(subset=[TCGA_COL]).copy()
print("dropping null TCGA_DESC:", df.shape)

print("\nclean df2")
df2 = df2.dropna(subset=[TCGA_COL]).copy()
print("dropping null TCGA_DESC:", df2.shape)

print("\n merging df + df2")

merge_keys = ["COSMIC_ID", "DRUG_ID"]

# Ensure merge keys exist in both
for k in merge_keys:
    if k not in df.columns:
        raise KeyError(f"'{k}' not found in df columns")
    if k not in df2.columns:
        raise KeyError(f"'{k}' not found in df2 columns")

merged_df = pd.merge(
    df,
    df2,
    on=merge_keys,
    how="inner",
    suffixes=("_df", "_df2")
)

print("Merged shape:", merged_df.shape)

print(list(merged_df.columns)[:25])

if f"{TARGET_COL}_df" in merged_df.columns:
    target_used = f"{TARGET_COL}_df"
else:
    target_used = TARGET_COL  # fallback

print("\nTarget column used:", target_used)


tcga_used = f"{TCGA_COL}_df" if f"{TCGA_COL}_df" in merged_df.columns else TCGA_COL

features = [tcga_used, "DRUG_ID"]

# Drop missing features
merged_clean = merged_df.dropna(subset=features + [target_used]).copy()
print("\nMerged clean shape (after dropping missing features/target):", merged_clean.shape)

# Encode categoricals (preprocessing only)
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_encoded = encoder.fit_transform(merged_clean[features])

print("Encoded X shape:", X_encoded.shape)

# Final outputs
X = merged_clean[features]
y = merged_clean[target_used]

print("X shape:", X.shape)
print("y shape:", y.shape)