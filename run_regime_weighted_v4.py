import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")

PRODUCTION_CONFIG_PATH = Path("astro_model_config.json")
V4_CONFIG_PATH = Path("astro_model_config_v4.json")

BASELINE_SUMMARY_PATH = DATA_DIR / "ml_model_summary_before_regime_weighted_v4.csv"
BASELINE_SELECTED_PATH = DATA_DIR / "selected_features_before_regime_weighted_v4.csv"

RESULTS_OUTPUT_PATH = DATA_DIR / "regime_weighted_results.csv"
REPORT_OUTPUT_PATH = DATA_DIR / "regime_weighted_report.md"

ML_SUMMARY_PATH = DATA_DIR / "ml_model_summary.csv"
SELECTED_FEATURES_PATH = DATA_DIR / "selected_features.csv"
RAW_ASPECTS_PATH = DATA_DIR / "astro_aspects_raw.csv"
DAILY_SCORE_PATH = DATA_DIR / "bitcoin_astro_daily_score.csv"


def run_step(description, command):
    print(f"\n=== {description} ===")
    subprocess.run(command, check=True)


def snapshot_file(source_path, dest_path):
    if not source_path.exists():
        raise FileNotFoundError(f"Missing required file: {source_path}")
    shutil.copy2(source_path, dest_path)


def load_csv(path, required_columns=None):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty")

    required_columns = required_columns or []
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")

    return df


def select_best_model(summary_df):
    ranked = summary_df.sort_values(
        ["balanced_score", "return_drawdown_ratio", "direction_accuracy"],
        ascending=[False, False, False],
    )
    return ranked.iloc[0]


def metric_row(stage, summary_df, selected_df):
    best = select_best_model(summary_df)
    return {
        "stage": stage,
        "best_horizon_days": int(best["horizon_days"]),
        "balanced_score": float(best["balanced_score"]),
        "return_drawdown_ratio": float(best["return_drawdown_ratio"]),
        "accuracy": float(best["direction_accuracy"]),
        "trades": int(best["number_of_trades"]),
        "selected_features": int(selected_df["feature"].dropna().astype(str).nunique()),
    }


def render_markdown_table(df):
    headers = list(df.columns)
    rows = [headers] + [[str(row[column]) for column in headers] for _, row in df.iterrows()]
    widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]

    def render_row(values):
        return "| " + " | ".join(
            values[index].ljust(widths[index]) for index in range(len(values))
        ) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    rendered = [render_row(rows[0]), separator]
    for row in rows[1:]:
        rendered.append(render_row(row))
    return "\n".join(rendered)


def write_outputs(production_row, v4_row):
    results = pd.DataFrame([production_row, v4_row])
    results["delta_balanced_score"] = results["balanced_score"].diff()
    results["delta_return_drawdown_ratio"] = results["return_drawdown_ratio"].diff()
    results["delta_accuracy"] = results["accuracy"].diff()
    results["delta_trades"] = results["trades"].diff()
    results["delta_selected_features"] = results["selected_features"].diff()
    results.to_csv(RESULTS_OUTPUT_PATH, index=False)

    display = results[[
        "stage",
        "best_horizon_days",
        "balanced_score",
        "return_drawdown_ratio",
        "accuracy",
        "trades",
        "selected_features",
    ]].copy()
    display["balanced_score"] = display["balanced_score"].map(lambda value: f"{value:.4f}")
    display["return_drawdown_ratio"] = display["return_drawdown_ratio"].map(lambda value: f"{value:.4f}")
    display["accuracy"] = display["accuracy"].map(lambda value: f"{value:.4f}")

    improved = "yes" if v4_row["balanced_score"] > production_row["balanced_score"] else "no"
    report_lines = [
        "# Regime-Weighted Astro Engine Report",
        "",
        "Production config was left unchanged. The v4 run used a separate regime-aware config.",
        "",
        f"- Production config: `{PRODUCTION_CONFIG_PATH}`",
        f"- Regime-aware config: `{V4_CONFIG_PATH}`",
        "",
        "## Production vs Regime-Aware Astro Engine",
        render_markdown_table(display),
        "",
        "## Interpretation",
        f"- Improved over production on balanced score: {improved}",
        f"- Balanced score delta: {v4_row['balanced_score'] - production_row['balanced_score']:.4f}",
        f"- Return/drawdown delta: {v4_row['return_drawdown_ratio'] - production_row['return_drawdown_ratio']:.4f}",
        f"- Accuracy delta: {v4_row['accuracy'] - production_row['accuracy']:.4f}",
        f"- Trade delta: {v4_row['trades'] - production_row['trades']}",
    ]

    REPORT_OUTPUT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main():
    if not RAW_ASPECTS_PATH.exists():
        raise FileNotFoundError(f"Missing raw aspects file: {RAW_ASPECTS_PATH}")

    python = sys.executable

    run_step(
        "Build Astro Engine v4 config",
        [python, "build_astro_model_config_v4.py"],
    )

    run_step(
        "Reconstruct production astro scores from production config",
        [
            python,
            "generate_astro_score.py",
            "--config",
            str(PRODUCTION_CONFIG_PATH),
            "--price-cache-path",
            str(DAILY_SCORE_PATH),
            "--reweight-existing-raw-path",
            str(RAW_ASPECTS_PATH),
        ],
    )
    run_step("Rebuild production ML dataset", [python, "build_ml_dataset.py"])
    run_step("Train production all-features ML", [python, "train_ml_model.py", "--feature-set", "all"])
    run_step("Analyze production feature stability", [python, "analyze_feature_stability.py"])
    run_step("Select production features", [python, "select_features.py"])
    run_step("Train production selected-feature ML", [python, "train_ml_model.py", "--feature-set", "selected"])

    snapshot_file(ML_SUMMARY_PATH, BASELINE_SUMMARY_PATH)
    snapshot_file(SELECTED_FEATURES_PATH, BASELINE_SELECTED_PATH)

    production_summary = load_csv(
        BASELINE_SUMMARY_PATH,
        ["horizon_days", "balanced_score", "return_drawdown_ratio", "direction_accuracy", "number_of_trades"],
    )
    production_selected = load_csv(BASELINE_SELECTED_PATH, ["feature"])
    production_row = metric_row("production", production_summary, production_selected)

    run_step(
        "Generate regime-aware astro scores",
        [
            python,
            "generate_astro_score.py",
            "--config",
            str(V4_CONFIG_PATH),
            "--price-cache-path",
            str(DAILY_SCORE_PATH),
            "--reweight-existing-raw-path",
            str(RAW_ASPECTS_PATH),
        ],
    )
    run_step("Rebuild regime-aware ML dataset", [python, "build_ml_dataset.py"])
    run_step("Train regime-aware all-features ML", [python, "train_ml_model.py", "--feature-set", "all"])
    run_step("Analyze regime-aware feature stability", [python, "analyze_feature_stability.py"])
    run_step("Select regime-aware features", [python, "select_features.py"])
    run_step("Train regime-aware selected-feature ML", [python, "train_ml_model.py", "--feature-set", "selected"])

    v4_summary = load_csv(
        ML_SUMMARY_PATH,
        ["horizon_days", "balanced_score", "return_drawdown_ratio", "direction_accuracy", "number_of_trades"],
    )
    v4_selected = load_csv(SELECTED_FEATURES_PATH, ["feature"])
    v4_row = metric_row("regime_aware_v4", v4_summary, v4_selected)

    write_outputs(production_row, v4_row)

    print(f"Wrote {RESULTS_OUTPUT_PATH}")
    print(f"Wrote {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
