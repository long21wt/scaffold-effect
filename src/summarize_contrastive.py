import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

plt.style.use(["science", "grid", "vibrant"])
plt.figure(dpi=300)
plt.close()

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.size": 20,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
        "figure.figsize": (16, 9),
        "text.latex.preamble": r"\usepackage{mathptmx}",
    }
)


def load_jsonl(path: str) -> pd.DataFrame:
    records = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    rows = []
    for r in records:
        rows.append(
            {
                "filename": r["filename"],
                "true_label": r.get("true_label", "mdd"),
                "case_type": r["case_type"],
                "base_pred": r["baseline"]["pred"],
                "full_pred": r["full"]["pred"],
                "base_p_mdd_norm": r["baseline"]["p_mdd_norm"],
                "full_p_mdd_norm": r["full"]["p_mdd_norm"],
                "base_p_mdd_seq": r["baseline"]["p_mdd_seq"],
                "full_p_mdd_seq": r["full"]["p_mdd_seq"],
                "base_decision_margin": r["baseline"]["decision_margin"],
                "full_decision_margin": r["full"]["decision_margin"],
                "delta_p_mdd": r["delta_p_mdd"],
            }
        )
    return pd.DataFrame(rows)


def infer_split(path: str) -> str:
    p = path.lower()
    for s in ("train", "dev", "test"):
        if (
            f"_{s}_" in p
            or p.endswith(f"_{s}.jsonl")
            or p.endswith(f"_{s}_contrastive.jsonl")
        ):
            return s
    return Path(path).stem


def infer_model(path: str) -> str:
    stem = Path(path).stem
    if stem.startswith("results_"):
        stem = stem[len("results_") :]
    if stem.endswith("_contrastive"):
        stem = stem[: -len("_contrastive")]
    for s in ("_train", "_dev", "_test"):
        if stem.endswith(s):
            stem = stem[: -len(s)]
            break
    for mode in ("_tabular_mri_preamble", "_tabular_parcel_mri", "_tabular"):
        if stem.endswith(mode):
            stem = stem[: -len(mode)]
            break
    return stem


def load_file_list(paths: list) -> pd.DataFrame:
    dfs = []
    for p in sorted(paths):
        df = load_jsonl(p)
        df["split"] = infer_split(p)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def build_three_way(df_preamble: pd.DataFrame, df_full: pd.DataFrame) -> pd.DataFrame:
    base = df_preamble[
        ["filename", "split", "true_label", "base_p_mdd_norm", "base_pred"]
    ].copy()
    base.columns = [
        "filename",
        "split",
        "true_label",
        "p_mdd_baseline",
        "pred_baseline",
    ]

    pre = df_preamble[
        ["filename", "split", "full_p_mdd_norm", "full_pred", "case_type"]
    ].copy()
    pre.columns = [
        "filename",
        "split",
        "p_mdd_preamble",
        "pred_preamble",
        "case_type_preamble",
    ]

    ful = df_full[
        ["filename", "split", "full_p_mdd_norm", "full_pred", "case_type"]
    ].copy()
    ful.columns = ["filename", "split", "p_mdd_full", "pred_full", "case_type_full"]

    merged = base.merge(pre, on=["filename", "split"], how="inner").merge(
        ful, on=["filename", "split"], how="inner"
    )

    merged["delta_preamble"] = merged["p_mdd_preamble"] - merged["p_mdd_baseline"]
    merged["delta_full"] = merged["p_mdd_full"] - merged["p_mdd_baseline"]
    merged["delta_full_vs_preamble"] = merged["p_mdd_full"] - merged["p_mdd_preamble"]

    merged["correct_baseline"] = merged["pred_baseline"].str.contains("Major")
    merged["correct_preamble"] = merged["pred_preamble"].str.contains("Major")
    merged["correct_full"] = merged["pred_full"].str.contains("Major")

    return merged


def summarize_three_way(df: pd.DataFrame, split: str) -> dict:
    n = len(df)

    def pct(mask: pd.Series) -> float:
        return 100 * mask.sum() / n if n else 0.0

    def m(col: str) -> float:
        return df[col].mean()

    def s(col: str) -> float:
        return df[col].std()

    def wilcoxon(a: pd.Series, b: pd.Series) -> tuple[float, float]:
        diff = a - b
        if (diff == 0).all():
            return float("nan"), float("nan")
        try:
            stat, p = stats.wilcoxon(diff.dropna(), alternative="greater")
            return float(stat), float(p)
        except ValueError:
            return float("nan"), float("nan")

    w_pre_stat, w_pre_p = wilcoxon(df["p_mdd_preamble"], df["p_mdd_baseline"])
    w_full_stat, w_full_p = wilcoxon(df["p_mdd_full"], df["p_mdd_baseline"])
    w_fvp_stat, w_fvp_p = wilcoxon(df["p_mdd_full"], df["p_mdd_preamble"])

    return {
        "split": split,
        "n_total": n,
        "acc_baseline": pct(df["correct_baseline"]),
        "acc_preamble": pct(df["correct_preamble"]),
        "acc_full": pct(df["correct_full"]),
        "mean_p_mdd_baseline": m("p_mdd_baseline"),
        "std_p_mdd_baseline": s("p_mdd_baseline"),
        "mean_p_mdd_preamble": m("p_mdd_preamble"),
        "std_p_mdd_preamble": s("p_mdd_preamble"),
        "mean_p_mdd_full": m("p_mdd_full"),
        "std_p_mdd_full": s("p_mdd_full"),
        "mean_delta_preamble": m("delta_preamble"),
        "std_delta_preamble": s("delta_preamble"),
        "mean_delta_full": m("delta_full"),
        "std_delta_full": s("delta_full"),
        "mean_delta_full_vs_pre": m("delta_full_vs_preamble"),
        "std_delta_full_vs_pre": s("delta_full_vs_preamble"),
        "wilcoxon_preamble_vs_base_stat": w_pre_stat,
        "wilcoxon_preamble_vs_base_p": w_pre_p,
        "wilcoxon_full_vs_base_stat": w_full_stat,
        "wilcoxon_full_vs_base_p": w_full_p,
        "wilcoxon_full_vs_preamble_stat": w_fvp_stat,
        "wilcoxon_full_vs_preamble_p": w_fvp_p,
    }


def print_summary(s: dict) -> None:
    logger.info("\n%s (n=%d)", s["split"].upper(), s["n_total"])
    logger.info(
        "accuracy: baseline=%.1f%% preamble=%.1f%% full=%.1f%%",
        s["acc_baseline"], s["acc_preamble"], s["acc_full"],
    )
    logger.info(
        "P(MDD): baseline=%.3f+-%.3f preamble=%.3f+-%.3f full=%.3f+-%.3f",
        s["mean_p_mdd_baseline"], s["std_p_mdd_baseline"],
        s["mean_p_mdd_preamble"], s["std_p_mdd_preamble"],
        s["mean_p_mdd_full"], s["std_p_mdd_full"],
    )
    logger.info(
        "delta vs base: preamble=%+.3f+-%.3f full=%+.3f+-%.3f",
        s["mean_delta_preamble"], s["std_delta_preamble"],
        s["mean_delta_full"], s["std_delta_full"],
    )
    logger.info(
        "delta full vs preamble: %+.3f+-%.3f",
        s["mean_delta_full_vs_pre"],
        s["std_delta_full_vs_pre"],
    )
    logger.info(
        "wilcoxon preamble>base: W=%.1f p=%.4f",
        s["wilcoxon_preamble_vs_base_stat"],
        s["wilcoxon_preamble_vs_base_p"],
    )
    logger.info(
        "wilcoxon full>base: W=%.1f p=%.4f",
        s["wilcoxon_full_vs_base_stat"],
        s["wilcoxon_full_vs_base_p"],
    )
    logger.info(
        "wilcoxon full>preamble: W=%.1f p=%.4f",
        s["wilcoxon_full_vs_preamble_stat"],
        s["wilcoxon_full_vs_preamble_p"],
    )


COND_LABELS = [
    r"$\textsc{text(arcf)}$",
    r"$\textsc{text(arcf)}$" + "\n" + r"$+ \textsc{prompt(MRI)}$",
    r"$\textsc{text(arcf, parcel)}$"
    + "\n"
    + r"$+ \textsc{prompt(MRI)}$"
    + "\n"
    + r"$+ \textsc{MRI\ plot}$",
]
COND_COLORS = ["#2196F3", "#FF9800", "#9C27B0"]
COND_KEYS = ["p_mdd_baseline", "p_mdd_preamble", "p_mdd_full"]


def plot_three_way_paired(
    df: pd.DataFrame, output_dir: str, model_name: str = ""
) -> None:
    """
    Figure 1: Group mean P(MDD) across three conditions with ±1 SD shading.
    """
    n = len(df)
    logger.info("P(MDD) across three conditions (n=%d)", n)

    _fig, ax = plt.subplots(figsize=(8, 4.5))

    xs = [0, 1, 2]
    means = [df[k].mean() for k in COND_KEYS]
    stds = [df[k].std() for k in COND_KEYS]

    # std shading
    ax.fill_between(
        xs,
        [m - s for m, s in zip(means, stds, strict=False)],
        [m + s for m, s in zip(means, stds, strict=False)],
        color="black",
        alpha=0.12,
        label=r"$\pm 1$ SD",
    )

    # mean line
    ax.plot(
        xs,
        means,
        color="black",
        linewidth=2.5,
        marker="o",
        markersize=7,
        zorder=6,
        label="Group mean",
    )

    # decision boundary
    ax.axhline(
        0.5,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.5,
        label="Decision boundary (0.5)",
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(COND_LABELS, fontsize=15)
    ax.set_ylabel(r"$\hat{P}(\mathrm{MDD})$", fontsize=15)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(-0.05, 1.05)

    ax.legend(
        fontsize=13,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        borderaxespad=0,
        frameon=True,
        columnspacing=1.2,
        handlelength=1.8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.78)
    fname = (
        f"fig1_three_way_paired_{model_name}.pdf"
        if model_name
        else "fig1_three_way_paired.pdf"
    )
    plt.savefig(
        Path(output_dir) / fname, dpi=300, bbox_inches="tight", pad_inches=0.02
    )
    plt.close()
    logger.info("saved %s", fname)


def save_summary_csv(summaries: list, output_dir: str, model_name: str = "") -> None:
    df = pd.DataFrame(summaries)
    fname = (
        f"summary_three_way_{model_name}.csv" if model_name else "summary_three_way.csv"
    )
    out = Path(output_dir) / fname
    df.to_csv(out, index=False, float_format="%.4f")
    logger.info("saved CSV: %s", out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--preamble_files",
        type=str,
        nargs="+",
        required=True,
        help="Contrastive JSONL files from tabular_mri_preamble mode",
    )
    p.add_argument(
        "--full_files",
        type=str,
        nargs="+",
        required=True,
        help="Contrastive JSONL files from tabular_parcel_mri mode",
    )
    p.add_argument("--output_dir", type=str, default="./summary")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model_name = infer_model(sorted(args.preamble_files)[0])
    log_path = Path(args.output_dir) / f"summary_three_way_{model_name}.log"

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    logger.info("model: %s", model_name)
    logger.info("preamble: %s", ", ".join(sorted(args.preamble_files)))
    logger.info("full: %s", ", ".join(sorted(args.full_files)))
    logger.info("date: %s", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

    df_preamble = load_file_list(args.preamble_files)
    df_full = load_file_list(args.full_files)

    summaries = []
    splits = sorted(set(df_preamble["split"].unique()) | set(df_full["split"].unique()))

    all_three_parts = []
    for split in splits:
        pre_s = df_preamble[df_preamble["split"] == split]
        ful_s = df_full[df_full["split"] == split]
        if pre_s.empty or ful_s.empty:
            logger.warning("missing data for split=%s, skipping", split)
            continue
        three = build_three_way(pre_s, ful_s)
        three["split"] = split
        all_three_parts.append(three)
        summaries.append(summarize_three_way(three, split))
        print_summary(summaries[-1])

    combined = pd.concat(all_three_parts, ignore_index=True)
    combined_summary = summarize_three_way(combined, "combined")
    summaries.append(combined_summary)
    print_summary(combined_summary)

    plot_three_way_paired(combined, args.output_dir, model_name)
    save_summary_csv(summaries, args.output_dir, model_name)

    logger.info("all outputs saved to %s", args.output_dir)
    file_handler.close()
    logger.removeHandler(file_handler)
    logger.info("log written to %s", log_path)
