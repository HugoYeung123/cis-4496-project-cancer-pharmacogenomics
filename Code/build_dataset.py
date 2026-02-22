"""
Build a clean "model table" for GDSC drug-response modeling.

Inputs (you already uploaded these):
- GDSC2-dataset.csv.zip  (screening rows with LN_IC50/AUC)
- Compounds-annotation.csv (drug annotations: targets/pathways)
- Cell_Lines_Details.xlsx (cell line metadata)

Outputs:
- model_table.parquet  (one row per screening measurement)
- schema.json          (column list + dtypes)
- missingness.csv      (missing rate per column)

USAGE
-----
python build_dataset.py \
  --gdsc_zip "/mnt/data/GDSC2-dataset.csv.zip" \
  --compounds_csv "/mnt/data/Compounds-annotation.csv" \
  --cell_lines_xlsx "/mnt/data/Cell_Lines_Details.xlsx" \
  --out_dir "/mnt/data/gdsc_ds_project/data"

Notes
-----
GDSC is cell-line screening data (research). This pipeline is for data-science coursework,
not for clinical decision-making without validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import pandas as pd


def read_zip_csv(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        csvs = [n for n in names if n.lower().endswith(".csv")]
        name = csvs[0] if csvs else names[0]
        with z.open(name) as f:
            return pd.read_csv(f)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Keep a consistent naming convention across variants
    ren = {
        "PUTATIVE_TARGET": "TARGET",
        "PATHWAY_NAME": "TARGET_PATHWAY",
        "Cancer Type (matching TCGA label)": "CANCER_TYPE_TCGA_MATCH",
        "Microsatellite instability Status (MSI)": "MSI_STATUS",
        "GDSC Tissue descriptor 1": "TISSUE_DESC_1",
        "GDSC Tissue descriptor 2": "TISSUE_DESC_2",
        "Screen Medium": "SCREEN_MEDIUM",
        "Growth Properties": "GROWTH_PROPERTIES",
    }
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gdsc_zip", type=str, required=True)
    ap.add_argument("--compounds_csv", type=str, required=True)
    ap.add_argument("--cell_lines_xlsx", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gdsc = read_zip_csv(Path(args.gdsc_zip))
    gdsc = normalize_columns(gdsc)

    compounds = pd.read_csv(args.compounds_csv)
    # avoid collisions + keep high-signal annotation columns if present
    compounds = compounds.rename(columns={"TARGET_PATHWAY": "TARGET_PATHWAY_ANNOT"})

    # Cell line metadata: read first sheet by default
    cell_lines = pd.read_excel(args.cell_lines_xlsx)
    cell_lines = normalize_columns(cell_lines)

    # Keys commonly available
    # GDSC2 usually has COSMIC_ID + CELL_LINE_NAME. We'll merge on COSMIC_ID if possible.
    if "COSMIC_ID" in gdsc.columns and "COSMIC_ID" in cell_lines.columns:
        merged = gdsc.merge(cell_lines, on="COSMIC_ID", how="left", suffixes=("", "_CL"))
        join_key = "COSMIC_ID"
    elif "CELL_LINE_NAME" in gdsc.columns and "CELL_LINE_NAME" in cell_lines.columns:
        merged = gdsc.merge(cell_lines, on="CELL_LINE_NAME", how="left", suffixes=("", "_CL"))
        join_key = "CELL_LINE_NAME"
    else:
        merged = gdsc.copy()
        join_key = None

    # Merge drug annotations
    if "DRUG_ID" in merged.columns and "DRUG_ID" in compounds.columns:
        merged = merged.merge(compounds, on="DRUG_ID", how="left", suffixes=("", "_DRUG"))
    else:
        # If DRUG_ID is missing, we still output the table
        pass

    # Minimal cleanups: enforce DRUG_ID as string categorical
    if "DRUG_ID" in merged.columns:
        merged["DRUG_ID"] = merged["DRUG_ID"].astype("Int64").astype("string")

    # Save model table
    # Parquet is fast and preserves types (better for DS workflows)
    out_parquet = out_dir / "model_table.parquet"
    merged.to_parquet(out_parquet, index=False)

    # Schema + missingness
    schema = {c: str(merged[c].dtype) for c in merged.columns}
    (out_dir / "schema.json").write_text(json.dumps({
        "n_rows": int(len(merged)),
        "n_cols": int(merged.shape[1]),
        "join_key": join_key,
        "columns": schema
    }, indent=2))

    miss = (merged.isna().mean().sort_values(ascending=False).reset_index())
    miss.columns = ["column", "missing_rate"]
    miss.to_csv(out_dir / "missingness.csv", index=False)

    print("Wrote:", out_parquet)
    print("Wrote:", out_dir / "schema.json")
    print("Wrote:", out_dir / "missingness.csv")


if __name__ == "__main__":
    main()
