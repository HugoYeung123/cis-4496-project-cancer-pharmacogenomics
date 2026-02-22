"""
DS-focused Streamlit dashboard:
- Doctor-facing ranking view (research)
- Data science panels: model metrics, leaderboard, global importance, coverage stats
- Evidence table: observed GDSC screening rows for selected cancer type + drug

Run:
streamlit run app_streamlit_ds.py
"""
from pathlib import Path
import zipfile
import json

import numpy as np
import pandas as pd
import streamlit as st
import joblib


@st.cache_data
def read_zip_csv(zip_path: str) -> pd.DataFrame:
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        csvs = [n for n in names if n.lower().endswith(".csv")]
        name = csvs[0] if csvs else names[0]
        with z.open(name) as f:
            return pd.read_csv(f)


@st.cache_data
def read_model_table(parquet_path: str) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)


@st.cache_resource
def load_artifacts(artifact_dir: str):
    ad = Path(artifact_dir)
    pipe = joblib.load(ad / "best_model.joblib")
    metrics = json.loads((ad / "metrics.json").read_text())
    feat = json.loads((ad / "feature_info.json").read_text())
    leaderboard = pd.read_csv(ad / "leaderboard.csv") if (ad / "leaderboard.csv").exists() else None
    permimp = pd.read_csv(ad / "permutation_importance.csv") if (ad / "permutation_importance.csv").exists() else None
    return pipe, metrics, feat, leaderboard, permimp


def main():
    st.set_page_config(page_title="GDSC DS Dashboard (Research)", layout="wide")
    st.title("GDSC Drug Response Dashboard — Data Science Prototype")
    st.caption("Cell-line screening decision-support prototype: modeling + evaluation + interpretability + evidence.")

    with st.sidebar:
        st.header("Paths")
        model_table = st.text_input("Model table parquet", value="gdsc_ds_project/data/model_table.parquet")
        artifact_dir = st.text_input("Artifacts dir", value="gdsc_ds_project/artifacts")
        target = st.selectbox("Target (lower is better)", ["LN_IC50", "AUC"], index=0)
        top_k = st.slider("Top K drugs", 5, 50, 15, 5)

    # Load data + artifacts
    try:
        df = read_model_table(model_table)
    except Exception as e:
        st.error(f"Could not read model table parquet: {e}")
        st.info("Run build_dataset.py first to generate model_table.parquet.")
        return

    if target not in df.columns:
        st.error(f"Target '{target}' not found in model table.")
        return

    df = df.dropna(subset=[target]).copy()

    try:
        pipe, metrics, feat, leaderboard, permimp = load_artifacts(artifact_dir)
    except Exception as e:
        st.error(f"Could not load artifacts: {e}")
        st.info("Run train_compare_models.py first to generate artifacts.")
        return

    feature_cols = feat["feature_cols"]

    # Patient profile (proxy)
    with st.sidebar:
        st.header("Patient / Tumor Profile (proxy)")
        def choose(col, label):
            if col in df.columns:
                vals = sorted([v for v in df[col].dropna().astype(str).unique().tolist()])
                return st.selectbox(label, vals[:300])
            return "(missing)"

        cancer = choose("TCGA_DESC", "Cancer type (TCGA_DESC)")
        msi = choose("MSI_STATUS", "MSI status")
        tissue1 = choose("TISSUE_DESC_1", "Tissue descriptor 1")
        tissue2 = choose("TISSUE_DESC_2", "Tissue descriptor 2")
        growth = choose("GROWTH_PROPERTIES", "Growth properties")
        medium = choose("SCREEN_MEDIUM", "Screen medium")

    # Create per-drug inference rows
    drug_cols = [c for c in ["DRUG_ID", "DRUG_NAME", "TARGET", "TARGET_PATHWAY"] if c in df.columns]
    drugs = df[drug_cols].dropna(subset=["DRUG_ID", "DRUG_NAME"]).drop_duplicates().copy()
    drugs["DRUG_ID"] = drugs["DRUG_ID"].astype("Int64").astype("string")

    rows = []
    for _, r in drugs.iterrows():
        row = {
            "TCGA_DESC": cancer,
            "MSI_STATUS": msi,
            "TISSUE_DESC_1": tissue1,
            "TISSUE_DESC_2": tissue2,
            "GROWTH_PROPERTIES": growth,
            "SCREEN_MEDIUM": medium,
            "TARGET": r.get("TARGET"),
            "TARGET_PATHWAY": r.get("TARGET_PATHWAY"),
            "DRUG_ID": r.get("DRUG_ID"),
            "DRUG_NAME": r.get("DRUG_NAME"),
        }
        rows.append(row)

    score_df = pd.DataFrame(rows).reindex(columns=feature_cols)
    if "DRUG_ID" in score_df.columns:
        score_df["DRUG_ID"] = score_df["DRUG_ID"].astype("string")

    preds = pipe.predict(score_df)
    out = pd.DataFrame(rows)
    out[f"{target}_PRED"] = preds

    # Evidence strength: how many screening rows exist for this cancer+drug in the dataset?
    ev = df.copy()
    if "TCGA_DESC" in ev.columns:
        ev = ev[ev["TCGA_DESC"].astype(str) == str(cancer)]
    counts = ev.groupby("DRUG_NAME").size().rename("N_EVIDENCE").reset_index() if "DRUG_NAME" in ev.columns else pd.DataFrame(columns=["DRUG_NAME","N_EVIDENCE"])
    out = out.merge(counts, on="DRUG_NAME", how="left")
    out["N_EVIDENCE"] = out["N_EVIDENCE"].fillna(0).astype(int)

    # Rank: primarily predicted sensitivity, secondarily evidence
    out = out.sort_values([f"{target}_PRED", "N_EVIDENCE"], ascending=[True, False]).head(top_k)

    # Layout
    tab1, tab2, tab3 = st.tabs(["Clinical view", "Model & metrics", "Data coverage"])

    with tab1:
        left, right = st.columns([1.15, 0.85])
        with left:
            st.subheader(f"Top {top_k} drugs (predicted lower = better)")
            st.dataframe(
                out[["DRUG_NAME","DRUG_ID","TARGET","TARGET_PATHWAY","N_EVIDENCE",f"{target}_PRED"]]
                .reset_index(drop=True),
                use_container_width=True,
                hide_index=True
            )

        with right:
            st.subheader("Drug details + evidence")
            pick = st.selectbox("Select drug", out["DRUG_NAME"].tolist())
            drow = out[out["DRUG_NAME"] == pick].iloc[0]

            st.markdown(
                f"**Drug:** {drow['DRUG_NAME']}  \n"
                f"**Target:** {drow.get('TARGET','')}  \n"
                f"**Pathway:** {drow.get('TARGET_PATHWAY','')}  \n"
                f"**Evidence rows (same cancer type):** {int(drow['N_EVIDENCE'])}"
            )
            st.metric(f"Predicted {target}", f"{float(drow[f'{target}_PRED']):.3f}")

            st.markdown("### Symptoms / side effects")
            st.info(
                "GDSC does not contain patient side-effect data. "
                "Connect this dashboard to a drug-label dataset (openFDA/FDA labels) to display common adverse reactions."
            )

            # Evidence table
            evid = df.copy()
            if "TCGA_DESC" in evid.columns:
                evid = evid[evid["TCGA_DESC"].astype(str) == str(cancer)]
            if "DRUG_NAME" in evid.columns:
                evid = evid[evid["DRUG_NAME"] == pick]
            cols = [c for c in ["CELL_LINE_NAME","COSMIC_ID",target,"AUC","Z_SCORE"] if c in evid.columns]
            if len(evid) == 0:
                st.write("No matching evidence rows found for this cancer type + drug.")
            else:
                st.write("Observed screening rows (top 25 most sensitive):")
                st.dataframe(
                    evid[cols].dropna(subset=[target]).sort_values(target, ascending=True).head(25),
                    use_container_width=True,
                    hide_index=True
                )

    with tab2:
        st.subheader("Holdout metrics (grouped split)")
        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE", f"{metrics['test_rmse']:.3f}")
        m2.metric("MAE", f"{metrics['test_mae']:.3f}")
        m3.metric("R²", f"{metrics['test_r2']:.3f}")
        st.caption(f"Best model: {metrics['best_model']} | Group column: {metrics['group_col']}")

        if leaderboard is not None:
            st.markdown("### Model comparison (Grouped CV)")
            st.dataframe(leaderboard, use_container_width=True, hide_index=True)

        if permimp is not None:
            st.markdown("### Global importance (Permutation Importance)")
            st.dataframe(permimp.head(25), use_container_width=True, hide_index=True)
        else:
            st.info("To generate global importance, run: python explain_model.py ...")

    with tab3:
        st.subheader("Coverage and data quality checks")
        st.write("These help you justify the model as a *data-science* deliverable (imbalance, missingness, support).")

        # Coverage counts
        if "TCGA_DESC" in df.columns:
            cov = df.groupby("TCGA_DESC").size().sort_values(ascending=False).head(25).reset_index()
            cov.columns = ["TCGA_DESC", "N_ROWS"]
            st.markdown("### Top cancer types by number of screening rows")
            st.dataframe(cov, use_container_width=True, hide_index=True)

        # Missingness snapshot for key fields
        key_cols = [c for c in ["TCGA_DESC","MSI_STATUS","TISSUE_DESC_1","TISSUE_DESC_2","TARGET","TARGET_PATHWAY","DRUG_NAME",target] if c in df.columns]
        miss = df[key_cols].isna().mean().sort_values(ascending=False).reset_index()
        miss.columns = ["column", "missing_rate"]
        st.markdown("### Missingness (selected columns)")
        st.dataframe(miss, use_container_width=True, hide_index=True)

    st.divider()
    st.warning(
        "Research prototype only. GDSC is cell-line data (not patient outcomes). "
        "Do not use for clinical decision-making without external validation and proper governance."
    )


if __name__ == "__main__":
    main()
