import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =============================================================================
# LOAD DATA
# =============================================================================

df  = pd.read_csv("GDSC_DATASET.csv")
df2 = pd.read_csv("genomic_features.csv")
df3 = pd.read_csv("model_list_latest.csv")
df4 = pd.read_csv("Model.csv")

# Standardise COSMIC_ID column names across all files
df2 = df2.rename(columns={"COSMIC ID": "COSMIC_ID"})
df4 = df4.rename(columns={"COSMICID": "COSMIC_ID"})

for temp_df in [df, df2, df3, df4]:
    temp_df["COSMIC_ID"] = pd.to_numeric(temp_df["COSMIC_ID"], errors="coerce")

print("Data loaded:")
print("  df shape :", df.shape)
print("  df2 shape:", df2.shape)
print("  df3 shape:", df3.shape)
print("  df4 shape:", df4.shape)


# =============================================================================
# BUILD HELPER TABLES
# =============================================================================

# df3 (model_list_latest.csv) — MSI, Growth Properties, cancer_type, tissue
df3_small = df3[[
    "COSMIC_ID", "model_name", "tissue", "cancer_type",
    "cancer_type_detail", "growth_properties", "msi_status", "sample_site"
]].copy()

# df4 (Model.csv) — Oncotree labels used in TCGA Stage 3
df4_small = df4[[
    "COSMIC_ID", "CellLineName", "StrippedCellLineName", "OncotreeLineage",
    "OncotreePrimaryDisease", "OncotreeSubtype", "GrowthPattern",
    "OnboardedMedia", "TissueOrigin"
]].copy()


# =============================================================================
# MSI AND GROWTH PROPERTIES FILL
# =============================================================================

# Start working copy
df_clean = df.copy()

# Merge df3 helper columns needed for filling
df_clean = df_clean.merge(
    df3_small[["COSMIC_ID", "msi_status", "growth_properties", "cancer_type"]],
    on="COSMIC_ID",
    how="left"
)

# Fill MSI from df3 where missing in df
df_clean["Microsatellite instability Status (MSI)"] = (
    df_clean["Microsatellite instability Status (MSI)"]
    .fillna(df_clean["msi_status"])
)

# Fill Growth Properties from df3 where missing in df
df_clean["Growth Properties"] = (
    df_clean["Growth Properties"]
    .fillna(df_clean["growth_properties"])
)

print("\nAfter MSI and Growth fill:")
print("  MSI missing         :", df_clean["Microsatellite instability Status (MSI)"].isna().sum())
print("  Growth Prop missing :", df_clean["Growth Properties"].isna().sum())


# =============================================================================
# GDSC TISSUE DESCRIPTOR FILLS (sourced from df2)
# =============================================================================

# Build df2 lookup — one clean row per COSMIC_ID with descriptor columns
df2_check = df2.copy()

cols_needed = ["COSMIC_ID", "TCGA Desc", "GDSC Desc1", "GDSC Desc2"]
df2_check = df2_check[[c for c in cols_needed if c in df2_check.columns]].copy()
df2_check["COSMIC_ID"] = pd.to_numeric(df2_check["COSMIC_ID"], errors="coerce")

for col in ["TCGA Desc", "GDSC Desc1", "GDSC Desc2"]:
    if col in df2_check.columns:
        df2_check[col] = (
            df2_check[col]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        )

df2_check = df2_check[df2_check["COSMIC_ID"].notna()].copy()


def first_non_null(series):
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) > 0 else pd.NA


df2_lookup = (
    df2_check.groupby("COSMIC_ID", as_index=False)
    .agg({
        "TCGA Desc": first_non_null,
        "GDSC Desc1": first_non_null,
        "GDSC Desc2": first_non_null
    })
)

# --- Fill GDSC Tissue descriptor 2 ---
df_clean = df_clean.merge(
    df2_lookup[["COSMIC_ID", "GDSC Desc2"]].rename(columns={
        "GDSC Desc2": "GDSC_Tissue_descriptor_2_df2"
    }),
    on="COSMIC_ID",
    how="left"
)

df_clean["descriptor2_was_imputed"] = (
    df_clean["GDSC Tissue descriptor 2"].isna() &
    df_clean["GDSC_Tissue_descriptor_2_df2"].notna()
)

df_clean["GDSC Tissue descriptor 2"] = (
    df_clean["GDSC Tissue descriptor 2"]
    .fillna(df_clean["GDSC_Tissue_descriptor_2_df2"])
)

# --- Fill GDSC Tissue descriptor 1 ---
df_clean = df_clean.merge(
    df2_lookup[["COSMIC_ID", "GDSC Desc1"]].rename(columns={
        "GDSC Desc1": "GDSC_Tissue_descriptor_1_df2"
    }),
    on="COSMIC_ID",
    how="left"
)

df_clean["descriptor1_was_imputed"] = (
    df_clean["GDSC Tissue descriptor 1"].isna() &
    df_clean["GDSC_Tissue_descriptor_1_df2"].notna()
)

df_clean["GDSC Tissue descriptor 1"] = (
    df_clean["GDSC Tissue descriptor 1"]
    .fillna(df_clean["GDSC_Tissue_descriptor_1_df2"])
)

print("\nAfter tissue descriptor fills:")
print("  GDSC Tissue descriptor 1 missing:", df_clean["GDSC Tissue descriptor 1"].isna().sum())
print("  GDSC Tissue descriptor 2 missing:", df_clean["GDSC Tissue descriptor 2"].isna().sum())


# =============================================================================
# CLEAN SELECTED FEATURE COLUMNS
# =============================================================================

text_cols_to_clean = [
    "TARGET",
    "DRUG_NAME",
    "TCGA_DESC",
    "Microsatellite instability Status (MSI)",
    "Growth Properties",
    "GDSC Tissue descriptor 1",
    "GDSC Tissue descriptor 2",
]

text_cols_to_clean = [col for col in text_cols_to_clean if col in df_clean.columns]

for col in text_cols_to_clean:
    df_clean[col] = (
        df_clean[col]
        .astype("string")
        .str.strip()
        .replace({
            "": pd.NA,
            " ": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
            "NONE": pd.NA,
            "null": pd.NA,
            "NULL": pd.NA,
        })
    )

print("\nAfter feature column cleaning — missing counts:")
print(df_clean[text_cols_to_clean].isna().sum().to_string())


# =============================================================================
# TCGA PREPROCESSING
# =============================================================================

# ----- Stage 1: Fill missing TCGA_DESC -----

df_clean["tcga_missing_before_stage1"] = df_clean["TCGA_DESC"].isna()
tcga_missing_before = int(df_clean["TCGA_DESC"].isna().sum())

# Pass 1: fill from "Cancer Type (matching TCGA label)" column
CANCER_TYPE_COL = "Cancer Type (matching TCGA label)"

if CANCER_TYPE_COL in df_clean.columns:
    df_clean[CANCER_TYPE_COL] = (
        df_clean[CANCER_TYPE_COL]
        .astype("string")
        .str.strip()
        .replace({
            "": pd.NA,
            "COAD/READ": "COREAD",
            "UNABLE TO CLASSIFY": "UNCLASSIFIED",
            "nan": pd.NA,
            "None": pd.NA,
        })
    )

    df_clean["tcga_filled_from_cancer_type"] = (
        df_clean["TCGA_DESC"].isna() & df_clean[CANCER_TYPE_COL].notna()
    )
    df_clean["TCGA_DESC"] = df_clean["TCGA_DESC"].fillna(df_clean[CANCER_TYPE_COL])
else:
    df_clean["tcga_filled_from_cancer_type"] = False

# Pass 2: manual cell-line remaps
cell_line_remap = {
    "MHH-CALL-4": "ALL",
    "BONNA-12":   "UNCLASSIFIED",
    "U-CH2":      "UNCLASSIFIED",
    "NCCIT":      "UNCLASSIFIED",
}

df_clean["tcga_filled_manually"] = False

for cell_line, new_label in cell_line_remap.items():
    mask = (
        (df_clean["CELL_LINE_NAME"] == cell_line) &
        df_clean["tcga_missing_before_stage1"] &
        df_clean["TCGA_DESC"].isna()
    )
    df_clean.loc[mask, "TCGA_DESC"] = new_label
    df_clean.loc[mask, "tcga_filled_manually"] = True

# Pass 3: fill from Tissue descriptor 2 only when the mapping is unique
t2_col = "GDSC Tissue descriptor 2"

t2_known = df_clean.loc[
    df_clean[t2_col].notna() & df_clean["TCGA_DESC"].notna(),
    [t2_col, "TCGA_DESC"]
].copy()

t2_uniques = (
    t2_known.groupby(t2_col)["TCGA_DESC"]
    .nunique()
    .reset_index(name="n_tcga")
)

t2_unique_only = t2_uniques.loc[t2_uniques["n_tcga"] == 1, t2_col]

t2_to_tcga_map = (
    t2_known[t2_known[t2_col].isin(t2_unique_only)]
    .drop_duplicates(subset=[t2_col, "TCGA_DESC"])
    .set_index(t2_col)["TCGA_DESC"]
    .to_dict()
)

df_clean["tcga_from_desc2_unique"] = df_clean[t2_col].map(t2_to_tcga_map)

df_clean["tcga_filled_from_desc2"] = (
    df_clean["tcga_missing_before_stage1"] &
    df_clean["TCGA_DESC"].isna() &
    df_clean["tcga_from_desc2_unique"].notna()
)

df_clean.loc[df_clean["tcga_filled_from_desc2"], "TCGA_DESC"] = (
    df_clean.loc[df_clean["tcga_filled_from_desc2"], "tcga_from_desc2_unique"]
)

tcga_missing_after = int(df_clean["TCGA_DESC"].isna().sum())

print("\nTCGA Stage 1 fill results:")
print("  Missing before                             :", tcga_missing_before)
print("  Filled from cancer-type column             :", int(df_clean["tcga_filled_from_cancer_type"].sum()))
print("  Filled manually                            :", int(df_clean["tcga_filled_manually"].sum()))
print("  Filled from Tissue descriptor 2 (unique)  :", int(df_clean["tcga_filled_from_desc2"].sum()))
print("  Missing after                              :", tcga_missing_after)

# ----- Stage 2: OTHER -> PRAD -----

other_before = int((df_clean["TCGA_DESC"] == "OTHER").sum())

df_clean["tcga_other_to_prad"] = df_clean["TCGA_DESC"] == "OTHER"
df_clean["TCGA_DESC"] = df_clean["TCGA_DESC"].replace({"OTHER": "PRAD"})

print("\nTCGA Stage 2 (OTHER -> PRAD):")
print("  Rows changed    :", other_before)
print("  OTHER remaining :", int((df_clean["TCGA_DESC"] == "OTHER").sum()))
print("  PRAD rows now   :", int((df_clean["TCGA_DESC"] == "PRAD").sum()))

# ----- Stage 3: UNCLASSIFIED resolution via Oncotree consensus -----

known_master = (
    df_clean.loc[
        df_clean["TCGA_DESC"].notna() &
        (df_clean["TCGA_DESC"] != "UNCLASSIFIED"),
        ["COSMIC_ID", "TCGA_DESC"]
    ]
    .drop_duplicates(subset="COSMIC_ID")
    .copy()
)

df4_oncotree = (
    df4[df4["COSMIC_ID"].notna()]
    [["COSMIC_ID", "OncotreeLineage", "OncotreePrimaryDisease", "OncotreeSubtype"]]
    .drop_duplicates(subset="COSMIC_ID")
)

df3_cancertype = (
    df3[df3["COSMIC_ID"].notna()]
    [["COSMIC_ID", "cancer_type"]]
    .drop_duplicates(subset="COSMIC_ID")
)

known_master = known_master.merge(df4_oncotree, on="COSMIC_ID", how="left")
known_master = known_master.merge(df3_cancertype, on="COSMIC_ID", how="left")


def build_unique_mapping(df_source, source_col, target_col="TCGA_DESC"):
    """Return a dict mapping source labels -> TCGA labels, only for 1-to-1 mappings."""
    tmp = df_source[[source_col, target_col]].dropna().copy()
    if tmp.empty:
        return {}
    tmp[source_col] = tmp[source_col].astype("string").str.strip()
    tmp[target_col] = tmp[target_col].astype("string").str.strip()
    tmp = tmp.loc[
        ~tmp[target_col].isin(["UNCLASSIFIED", "OTHER", "", "nan", "None"])
    ].copy()
    if tmp.empty:
        return {}
    counts = (
        tmp.groupby(source_col)[target_col]
        .nunique()
        .reset_index(name="n_tcga")
    )
    unique_keys = counts.loc[counts["n_tcga"] == 1, source_col]
    mapping = (
        tmp[tmp[source_col].isin(unique_keys)]
        .drop_duplicates(subset=[source_col, target_col])
        .set_index(source_col)[target_col]
        .to_dict()
    )
    return mapping


mapping_sources = ["cancer_type", "OncotreePrimaryDisease", "OncotreeSubtype", "OncotreeLineage"]
mapping_sources = [c for c in mapping_sources if c in known_master.columns]

helper_to_tcga_maps = {
    src: build_unique_mapping(known_master, src) for src in mapping_sources
}

cosmic_master = (
    df_clean.drop_duplicates(subset="COSMIC_ID")
    [["COSMIC_ID", "TCGA_DESC"]]
    .copy()
)
cosmic_master = cosmic_master.merge(df4_oncotree, on="COSMIC_ID", how="left")
cosmic_master = cosmic_master.merge(df3_cancertype, on="COSMIC_ID", how="left")

unclassified_master = cosmic_master.loc[
    cosmic_master["TCGA_DESC"] == "UNCLASSIFIED"
].copy()

for source_col, mp in helper_to_tcga_maps.items():
    pred_col = f"{source_col}__pred_tcga"
    if source_col in unclassified_master.columns:
        unclassified_master[pred_col] = (
            unclassified_master[source_col]
            .astype("string")
            .str.strip()
            .map(mp)
        )

pred_cols = [c for c in unclassified_master.columns if c.endswith("__pred_tcga")]


def consensus_tcga(row):
    vals = [str(v).strip() for v in row if pd.notna(v)]
    vals = [v for v in vals if v not in ("", "nan", "None")]
    unique_vals = sorted(set(vals))
    return unique_vals[0] if len(unique_vals) == 1 else pd.NA


unclassified_master["stage3_consensus_tcga"] = (
    unclassified_master[pred_cols].apply(consensus_tcga, axis=1)
)

resolved_cosmic = (
    unclassified_master.loc[
        unclassified_master["stage3_consensus_tcga"].notna(),
        ["COSMIC_ID", "stage3_consensus_tcga"]
    ]
    .drop_duplicates()
)

df_clean = df_clean.merge(
    resolved_cosmic.rename(columns={"stage3_consensus_tcga": "stage3_resolved_tcga"}),
    on="COSMIC_ID",
    how="left"
)

df_clean["tcga_resolved_from_stage3"] = (
    (df_clean["TCGA_DESC"] == "UNCLASSIFIED") &
    df_clean["stage3_resolved_tcga"].notna()
)

df_clean.loc[df_clean["tcga_resolved_from_stage3"], "TCGA_DESC"] = (
    df_clean.loc[df_clean["tcga_resolved_from_stage3"], "stage3_resolved_tcga"]
)

print("\nTCGA Stage 3 (UNCLASSIFIED resolution):")
print("  Rows resolved       :", int(df_clean["tcga_resolved_from_stage3"].sum()))
print("  UNCLASSIFIED remain :", int((df_clean["TCGA_DESC"] == "UNCLASSIFIED").sum()))
print("  TCGA_DESC missing   :", df_clean["TCGA_DESC"].isna().sum())


# =============================================================================
# MERGE GENOMIC FEATURES FROM df3 (model_list_latest.csv)
# =============================================================================

genomic_bio_candidates = [
    "COSMIC_ID", "ploidy_snp6", "ploidy_wes", "ploidy_wgs", "mutational_burden",
    "mlh1_expression_by_ihc", "mlh1_promoter_methylation_status",
    "msh2_expression_by_ihc", "pms2_expression_by_ihc", "msh6_expression_by_ihc",
    "braf_mutation_identified", "braf_expression_by_ihc", "pik3ca_mutation_identified",
    "pten_expression_by_ihc", "pten_mutation_identified", "kras_mutation_identified",
    "mismatch_repair_status",
]

gf_bio_cols_present = [c for c in genomic_bio_candidates if c in df3.columns]
gf_bio = df3[df3["COSMIC_ID"].notna()][gf_bio_cols_present].copy()
gf_bio_lookup = gf_bio.groupby("COSMIC_ID", as_index=False).agg(first_non_null)

df_gf_test = df_clean.merge(gf_bio_lookup, on="COSMIC_ID", how="left")
added_cols = [c for c in gf_bio_lookup.columns if c != "COSMIC_ID"]

coverage_pct = (df_gf_test[added_cols].notna().mean() * 100).sort_values(ascending=False)
min_coverage_pct = 20
genomic_cols_keep = coverage_pct[coverage_pct >= min_coverage_pct].index.tolist()

gf_bio_final = gf_bio_lookup[["COSMIC_ID"] + genomic_cols_keep].copy()

before_rows = len(df_clean)
df_clean = df_clean.merge(gf_bio_final, on="COSMIC_ID", how="left")
after_rows = len(df_clean)

print("\nGenomic feature merge:")
print("  Columns kept :", genomic_cols_keep)
print("  Rows before  :", before_rows, "| Rows after:", after_rows)
print("  Missing counts after merge:")
print(df_clean[genomic_cols_keep].isna().sum().to_string())


# =============================================================================
# FINAL HELPER-COLUMN CLEANUP
# =============================================================================

helper_cols_to_drop = [
    "msi_status",
    "growth_properties",
    "cancer_type",
    "GDSC_Tissue_descriptor_1_df2",
    "GDSC_Tissue_descriptor_2_df2",
    "TCGA_DESC_df2",
    "tcga_from_desc2_unique",
    "stage3_resolved_tcga",
    "descriptor1_was_imputed",
    "descriptor2_was_imputed",
    "tcga_missing_before_stage1",
    "tcga_filled_from_cancer_type",
    "tcga_filled_manually",
    "tcga_filled_from_desc2",
    "tcga_other_to_prad",
    "tcga_stage2_exception_to_unclassified",
    "tcga_resolved_from_stage3",
]

helper_cols_to_drop = [col for col in helper_cols_to_drop if col in df_clean.columns]
df_clean = df_clean.drop(columns=helper_cols_to_drop)

key_check_cols = [
    "LN_IC50", "DRUG_NAME", "TARGET", "TCGA_DESC",
    "Microsatellite instability Status (MSI)",
    "Growth Properties",
    "GDSC Tissue descriptor 1",
    "GDSC Tissue descriptor 2",
    "mutational_burden", "ploidy_snp6", "ploidy_wes",
]
key_check_cols = [col for col in key_check_cols if col in df_clean.columns]

print("\nFinal missing values after helper cleanup:")
print(df_clean[key_check_cols].isna().sum().to_string())
print("df_clean shape:", df_clean.shape)


# =============================================================================
# CHOOSE FINAL FEATURE SET
# =============================================================================

target_col = "LN_IC50"

id_cols = ["COSMIC_ID", "CELL_LINE_NAME", "DRUG_ID"]
id_cols = [col for col in id_cols if col in df_clean.columns]

candidate_feature_cols = [
    "DRUG_NAME",
    "TARGET",
    "TARGET_PATHWAY",
    "TCGA_DESC",
    "Microsatellite instability Status (MSI)",
    "Growth Properties",
    "GDSC Tissue descriptor 1",
    "GDSC Tissue descriptor 2",
    "mutational_burden",
    "ploidy_snp6",
    "ploidy_wes",
]
candidate_feature_cols = [col for col in candidate_feature_cols if col in df_clean.columns]

X_candidate = df_clean[candidate_feature_cols].copy()
y = df_clean[target_col].copy()
modeling_df = df_clean[id_cols + candidate_feature_cols + [target_col]].copy()

print("\nFinal feature set:")
print("  Target column     :", target_col)
print("  Candidate features:", candidate_feature_cols)
print("  X_candidate shape :", X_candidate.shape)
print("  y shape           :", y.shape)
print("  modeling_df shape :", modeling_df.shape)
print("\nMissing values in candidate features:")
print(X_candidate.isna().sum().sort_values(ascending=False).to_string())


# =============================================================================
# SAVE — PRE-ENCODING SNAPSHOT
# Saved here: all cleaning and filling is done, no encoding or scaling applied yet.
# This is the human-readable version that your group can inspect and share.
# =============================================================================

df_clean.to_csv("FINAL_PREPROCESSED_DF.csv", index=False)
print("\nSaved: FINAL_PREPROCESSED_DF.csv —", df_clean.shape)


# =============================================================================
# TRAIN / TEST SPLIT
# Done before encoding and scaling so transformations are learned from train only.
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_candidate,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTrain/test split complete:")
print("  X_train shape:", X_train.shape)
print("  X_test shape :", X_test.shape)
print("  y_train mean :", round(y_train.mean(), 4), "| std:", round(y_train.std(), 4))
print("  y_test mean  :", round(y_test.mean(), 4),  "| std:", round(y_test.std(), 4))


# =============================================================================
# ONE-HOT ENCODING
# =============================================================================

X_train = X_train.copy()
X_test  = X_test.copy()

# Columns to one-hot encode
categorical_cols = [
    "DRUG_NAME",
    "TCGA_DESC",
    "GDSC Tissue descriptor 1",
    "GDSC Tissue descriptor 2",
    "Cancer Type (matching TCGA label)",
    "Microsatellite instability Status (MSI)",
    "Screen Medium",
    "Growth Properties",
    "TARGET",
    "TARGET_PATHWAY",
]
categorical_cols = [col for col in categorical_cols if col in X_train.columns]

# Everything else is treated as numeric
numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

# Clean categorical columns: cast to string, fill missing with explicit label
for col in categorical_cols:
    X_train[col] = X_train[col].astype("string").fillna("MISSING")
    X_test[col]  = X_test[col].astype("string").fillna("MISSING")

# Clean numeric columns: force numeric, fill missing with training medians
for col in numeric_cols:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_test[col]  = pd.to_numeric(X_test[col],  errors="coerce")

train_medians = X_train[numeric_cols].median()
X_train[numeric_cols] = X_train[numeric_cols].fillna(train_medians)
X_test[numeric_cols]  = X_test[numeric_cols].fillna(train_medians)

# One-hot encode
X_train_cat = pd.get_dummies(X_train[categorical_cols], columns=categorical_cols, dummy_na=False)
X_test_cat  = pd.get_dummies(X_test[categorical_cols],  columns=categorical_cols, dummy_na=False)

# Align test columns to training columns (adds any missing OHE cols, fills with 0)
X_train_cat, X_test_cat = X_train_cat.align(X_test_cat, join="left", axis=1, fill_value=0)

# Combine encoded categorical + numeric columns
X_train_encoded = pd.concat([X_train_cat, X_train[numeric_cols]], axis=1)
X_test_encoded  = pd.concat([X_test_cat,  X_test[numeric_cols]],  axis=1)

print("\nOne-hot encoding complete:")
print("  X_train_encoded shape:", X_train_encoded.shape)
print("  X_test_encoded shape :", X_test_encoded.shape)


# =============================================================================
# STANDARDIZATION / FEATURE SCALING
# Scaler is fit on training data only — no data leakage.
# Only continuous numeric columns are scaled; OHE columns are left unchanged.
# =============================================================================

numeric_cols_to_scale = ["mutational_burden", "ploidy_snp6", "ploidy_wes"]
numeric_cols_to_scale = [col for col in numeric_cols_to_scale if col in X_train_encoded.columns]

X_train_scaled = X_train_encoded.copy()
X_test_scaled  = X_test_encoded.copy()

scaler = StandardScaler()
X_train_scaled[numeric_cols_to_scale] = scaler.fit_transform(
    X_train_encoded[numeric_cols_to_scale]
)
X_test_scaled[numeric_cols_to_scale] = scaler.transform(
    X_test_encoded[numeric_cols_to_scale]
)

print("\nStandardization complete:")
print("  Scaled columns      :", numeric_cols_to_scale)
print("  X_train_scaled shape:", X_train_scaled.shape)
print("  X_test_scaled shape :", X_test_scaled.shape)


# =============================================================================
# SAVE — POST-STANDARDIZATION SNAPSHOTS
# Train and test saved separately so they stay leakage-free.
# =============================================================================

X_train_scaled.to_csv("X_train_scaled.csv", index=False)
X_test_scaled.to_csv("X_test_scaled.csv",   index=False)

print("\nSaved: X_train_scaled.csv —", X_train_scaled.shape)
print("Saved: X_test_scaled.csv  —",  X_test_scaled.shape)