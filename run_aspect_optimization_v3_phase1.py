import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")

BASELINE_SUMMARY_PATH = DATA_DIR / "ml_model_summary_before_aspect_optimization_v3_phase1.csv"
BASELINE_SELECTED_PATH = DATA_DIR / "selected_features_before_aspect_optimization_v3_phase1.csv"

RESULTS_OUTPUT_PATH = DATA_DIR / "aspect_optimization_results.csv"
REPORT_OUTPUT_PATH = DATA_DIR / "aspect_optimization_report.md"

PRODUCTION_CONFIG_PATH = Path("astro_model_config.json")
EXPERIMENTAL_CONFIG_PATH = Path("astro_model_config_experimental.json")

ML_SUMMARY_PATH = DATA_DIR / "ml_model_summary.csv"
SELECTED_FEATURES_PATH = DATA_DIR / "selected_features.csv"


def run_step(description, command):
    print(f"\n=== {description} ===")
    subprocess.run(command, check=True)


def snapshot_file(source_path, dest_path):
    if not source_path.exists():
        raise FileNotFoundError(f"Missing required baseline file: {source_path}")
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


def count_selected_aspect_features(selected_df):
    features = selected_df["feature"].dropna().astype(str).tolist()
    aspect_features = sorted(
        feature
        for feature in features
        if feature.startswith("aspect_count_")
        or feature in {
            "conjunction_strength",
            "trine_strength",
            "sextile_strength",
            "square_strength",
            "opposition_strength",
        }
    )
    return aspect_features


def metric_row(stage, summary_df, selected_df):
    best = select_best_model(summary_df)
    aspect_features = count_selected_aspect_features(selected_df)
    return {
        "stage": stage,
        "best_horizon_days": int(best["horizon_days"]),
        "balanced_score": float(best["balanced_score"]),
        "return_drawdown_ratio": float(best["return_drawdown_ratio"]),
        "accuracy": float(best["direction_accuracy"]),
        "trades": int(best["number_of_trades"]),
        "selected_features": int(selected_df["feature"].dropna().astype(str).nunique()),
        "selected_aspect_features": int(len(aspect_features)),
        "selected_aspect_feature_list": ", ".join(aspect_features) if aspect_features else "none",
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


def write_outputs(production_row, optimized_row):
    results = pd.DataFrame([production_row, optimized_row])
    results["delta_balanced_score"] = results["balanced_score"].diff()
    results["delta_return_drawdown_ratio"] = results["return_drawdown_ratio"].diff()
    results["delta_accuracy"] = results["accuracy"].diff()
    results["delta_selected_features"] = results["selected_features"].diff()
    results["delta_selected_aspect_features"] = results["selected_aspect_features"].diff()
    results["delta_trades"] = results["trades"].diff()
    results.to_csv(RESULTS_OUTPUT_PATH, index=False)

    display = results[[
        "stage",
        "best_horizon_days",
        "balanced_score",
        "return_drawdown_ratio",
        "accuracy",
        "selected_features",
        "selected_aspect_features",
        "trades",
    ]].copy()
    display["balanced_score"] = display["balanced_score"].map(lambda value: f"{value:.4f}")
    display["return_drawdown_ratio"] = display["return_drawdown_ratio"].map(lambda value: f"{value:.4f}")
    display["accuracy"] = display["accuracy"].map(lambda value: f"{value:.4f}")

    report_lines = [
        "# Aspect Optimization Report",
        "",
        "Aspect-only experimental config was used. Planet weights and natal target weights were left unchanged.",
        "",
        f"- Production config: `{PRODUCTION_CONFIG_PATH}`",
        f"- Experimental config: `{EXPERIMENTAL_CONFIG_PATH}`",
        "",
        "## Production vs Aspect Optimized",
        render_markdown_table(display),
        "",
        "## Aspect Feature Retention",
        f"- Production selected aspect features: {production_row['selected_aspect_feature_list']}",
        f"- Optimized selected aspect features: {optimized_row['selected_aspect_feature_list']}",
        "",
        "## Interpretation",
        "- Use the optimized run only as evidence for whether discovered aspect weights improve the ML research pipeline.",
        "- No production config was overwritten in this phase.",
    ]

    REPORT_OUTPUT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main():
    if not EXPERIMENTAL_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing experimental config: {EXPERIMENTAL_CONFIG_PATH}"
        )

    snapshot_file(ML_SUMMARY_PATH, BASELINE_SUMMARY_PATH)
    snapshot_file(SELECTED_FEATURES_PATH, BASELINE_SELECTED_PATH)

    production_summary = load_csv(
        BASELINE_SUMMARY_PATH,
        ["horizon_days", "balanced_score", "return_drawdown_ratio", "direction_accuracy", "number_of_trades"],
    )
    production_selected = load_csv(BASELINE_SELECTED_PATH, ["feature"])
    production_row = metric_row("production", production_summary, production_selected)

    python = sys.executable

    run_step(
        "Generate astro scores using experimental aspect weights",
        [
            python,
            "generate_astro_score.py",
            "--config",
            str(EXPERIMENTAL_CONFIG_PATH),
            "--price-cache-path",
            "data/bitcoin_astro_daily_score.csv",
            "--reweight-existing-raw-path",
            "data/astro_aspects_raw.csv",
        ],
    )
    run_step("Rebuild ML dataset", [python, "build_ml_dataset.py"])
    run_step("Train ML model with all features", [python, "train_ml_model.py", "--feature-set", "all"])
    run_step("Analyze feature stability", [python, "analyze_feature_stability.py"])
    run_step("Select features", [python, "select_features.py"])
    run_step("Train ML model with selected features", [python, "train_ml_model.py", "--feature-set", "selected"])

    optimized_summary = load_csv(
        ML_SUMMARY_PATH,
        ["horizon_days", "balanced_score", "return_drawdown_ratio", "direction_accuracy", "number_of_trades"],
    )
    optimized_selected = load_csv(SELECTED_FEATURES_PATH, ["feature"])
    optimized_row = metric_row("aspect_optimized", optimized_summary, optimized_selected)

    write_outputs(production_row, optimized_row)

    print(f"Wrote {RESULTS_OUTPUT_PATH}")
    print(f"Wrote {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
