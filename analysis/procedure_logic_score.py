import argparse
import numpy as np
import pandas as pd

try:
    import krippendorff
except Exception:
    krippendorff = None  # type: ignore

# Weight for combining polarity agreement and confidence-within-polarity agreement.
# Typical choice: 0.7 (polarity matters more than confidence).
W = 0.7


def procedure_logic_agreement(df: pd.DataFrame, w: float = 0.7) -> dict:
    """
    Compute Procedure Logic agreement as:
      A = w * alpha_polarity + (1-w) * alpha_conf_within_polarity

    Where:
      - alpha_polarity: Krippendorff's alpha (nominal) on Yes/No answers
      - alpha_conf_within_polarity: weighted combination of:
          alpha_yes: alpha (ordinal) on confidence among YES answers
          alpha_no : alpha (ordinal) on confidence among NO answers
        weighted by number of available confidence ratings in each group.
    """
    df = df.copy()

    # --- Normalize and map answers to polarity (Yes=1, No=0) ---
    df["answer_norm"] = df["answer"].astype(str).str.strip().str.lower()
    map_pol = {"yes": 1, "y": 1, "да": 1, "no": 0, "n": 0, "нет": 0}
    df["polarity"] = df["answer_norm"].map(map_pol)

    # Convert confidence to numeric (coerce invalid values to NaN)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    # --- Polarity agreement: annotators x items matrix ---
    pol_mat = df.pivot_table(
        index="annotator_id",
        columns="item_id",
        values="polarity",
        aggfunc="first",
    )

    alpha_pol = krippendorff.alpha(
        reliability_data=pol_mat.to_numpy(dtype=float),
        level_of_measurement="nominal",
    )

    # --- Confidence agreement within YES / within NO ---
    conf_yes = df.where(df["polarity"] == 1).pivot_table(
        index="annotator_id",
        columns="item_id",
        values="confidence",
        aggfunc="first",
    )
    conf_no = df.where(df["polarity"] == 0).pivot_table(
        index="annotator_id",
        columns="item_id",
        values="confidence",
        aggfunc="first",
    )

    yes_arr = conf_yes.to_numpy(dtype=float)
    no_arr = conf_no.to_numpy(dtype=float)

    n_yes = int(np.isfinite(yes_arr).sum())
    n_no = int(np.isfinite(no_arr).sum())

    alpha_yes = (
        krippendorff.alpha(reliability_data=yes_arr, level_of_measurement="ordinal")
        if n_yes >= 2
        else np.nan
    )
    alpha_no = (
        krippendorff.alpha(reliability_data=no_arr, level_of_measurement="ordinal")
        if n_no >= 2
        else np.nan
    )

    # Combine YES/NO confidence agreement using available-rating weights
    parts = []
    if np.isfinite(alpha_yes) and n_yes > 0:
        parts.append((n_yes, alpha_yes))
    if np.isfinite(alpha_no) and n_no > 0:
        parts.append((n_no, alpha_no))

    alpha_conf = (
        sum(n * a for n, a in parts) / sum(n for n, _ in parts)
        if parts
        else np.nan
    )

    # Final combined agreement
    A = (
        w * alpha_pol + (1.0 - w) * alpha_conf
        if np.isfinite(alpha_pol) and np.isfinite(alpha_conf)
        else np.nan
    )

    return {
        "alpha_polarity_nominal": float(alpha_pol) if np.isfinite(alpha_pol) else np.nan,
        "alpha_conf_yes_ordinal": float(alpha_yes) if np.isfinite(alpha_yes) else np.nan,
        "alpha_conf_no_ordinal": float(alpha_no) if np.isfinite(alpha_no) else np.nan,
        "alpha_conf_combined": float(alpha_conf) if np.isfinite(alpha_conf) else np.nan,
        "procedure_logic_agreement_A": float(A) if np.isfinite(A) else np.nan,
        "w": float(w),
        "n_conf_yes": n_yes,
        "n_conf_no": n_no,
        "n_rows": int(len(df)),
        "n_items": int(df["item_id"].nunique()),
        "n_annotators": int(df["annotator_id"].nunique()),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute Procedure Logic agreement.")
    parser.add_argument("--csv", required=True, help="CSV with dataset_id, video_id, item_id, annotator_id, answer, confidence.")
    parser.add_argument("--w", type=float, default=W, help="Weight for polarity agreement.")
    args = parser.parse_args()

    # Read CSV
    df = pd.read_csv(args.csv)

    # Basic sanity checks / required columns
    required = {"dataset_id", "video_id", "item_id", "annotator_id", "answer", "confidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    # Drop rows without item_id or annotator_id (cannot be used)
    df = df.dropna(subset=["item_id", "annotator_id"])

    # Compute overall agreement
    overall = procedure_logic_agreement(df, w=args.w)

    print("=== Procedure Logic Agreement (OVERALL) ===")
    for k, v in overall.items():
        print(f"{k}: {v}")
    print()

    # Compute per (dataset_id, video_id)
    print("=== Procedure Logic Agreement by (dataset_id, video_id) ===")
    group_rows = []
    for (dataset_id, video_id), g in df.groupby(["dataset_id", "video_id"], dropna=False):
        res = procedure_logic_agreement(g, w=W)
        res["dataset_id"] = dataset_id
        res["video_id"] = video_id
        group_rows.append(res)

    per_video = pd.DataFrame(group_rows).sort_values(
        by=["dataset_id", "video_id"], kind="stable"
    )

    # Print a compact table
    cols = [
        "dataset_id", "video_id",
        "n_rows", "n_items", "n_annotators",
        "alpha_polarity_nominal",
        "alpha_conf_yes_ordinal", "alpha_conf_no_ordinal",
        "alpha_conf_combined",
        "procedure_logic_agreement_A",
    ]
    with pd.option_context("display.max_rows", 200, "display.max_columns", 50, "display.width", 140):
        print(per_video[cols].to_string(index=False))


if __name__ == "__main__":
    main()
    if krippendorff is None:
        raise RuntimeError("krippendorff package is not available.")
