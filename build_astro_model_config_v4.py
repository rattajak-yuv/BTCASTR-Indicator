import json
from pathlib import Path

import pandas as pd

BASE_CONFIG_PATH = Path("astro_model_config.json")
PLANET_INPUT_PATH = Path("data/regime_planet_importance.csv")
ASPECT_INPUT_PATH = Path("data/regime_aspect_importance.csv")
NATAL_INPUT_PATH = Path("data/regime_natal_importance.csv")
SUMMARY_INPUT_PATH = Path("data/regime_astro_summary.md")
OUTPUT_PATH = Path("astro_model_config_v4.json")

MARKET_REGIMES = ["Bull Market", "Bear Market", "Sideways"]
VOLATILITY_STATES = ["High Volatility", "Low Volatility"]


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


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def normalize_multiplier(row, lower_cap, upper_cap):
    relative_strength = float(row.get("relative_regime_strength", 1.0))
    confidence = clamp(float(row.get("confidence_score", 0.0)) / 100.0, 0.0, 0.85)
    recommendation = str(row.get("dynamic_weight_recommendation", "keep near global")).strip().lower()

    if recommendation == "insufficient evidence":
        return 1.0

    adjustment = (relative_strength - 1.0) * max(confidence, 0.20) * 0.45
    multiplier = 1.0 + adjustment

    if recommendation == "increase in this regime":
        multiplier = max(multiplier, 1.03)
    elif recommendation == "decrease in this regime":
        multiplier = min(multiplier, 0.97)
    elif abs(relative_strength - 1.0) < 0.10:
        multiplier = 1.0

    return round(clamp(multiplier, lower_cap, upper_cap), 6)


def build_profile(df, regime_name, component_column, lower_cap, upper_cap):
    subset = df[df["regime"] == regime_name].copy()
    profile = {}
    for _, row in subset.iterrows():
        component_name = str(row[component_column]).strip()
        profile[component_name] = normalize_multiplier(row, lower_cap, upper_cap)
    return profile


def build_dynamic_profiles(planet_df, aspect_df, natal_df):
    market_profiles = {}
    for regime_name in MARKET_REGIMES:
        market_profiles[regime_name] = {
            "planets": build_profile(planet_df, regime_name, "component_name", 0.85, 1.25),
            "aspects": build_profile(aspect_df, regime_name, "component_name", 0.80, 1.25),
            "natal_targets": build_profile(natal_df, regime_name, "component_name", 0.85, 1.20),
        }

    volatility_profiles = {}
    for state_name in VOLATILITY_STATES:
        volatility_profiles[state_name] = {
            "planets": build_profile(planet_df, state_name, "component_name", 0.90, 1.18),
            "aspects": build_profile(aspect_df, state_name, "component_name", 0.88, 1.20),
            "natal_targets": build_profile(natal_df, state_name, "component_name", 0.90, 1.18),
        }

    return {
        "market_regimes": market_profiles,
        "volatility_states": volatility_profiles,
        "volatility_overlay_strength": 0.60,
        "caps": {
            "planets": [0.75, 1.35],
            "aspects": [0.75, 1.35],
            "natal_targets": [0.75, 1.30],
        },
    }


def main():
    with BASE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        base_config = json.load(handle)

    planet_df = load_csv(
        PLANET_INPUT_PATH,
        ["regime", "component_name", "relative_regime_strength", "confidence_score", "dynamic_weight_recommendation"],
    )
    aspect_df = load_csv(
        ASPECT_INPUT_PATH,
        ["regime", "component_name", "relative_regime_strength", "confidence_score", "dynamic_weight_recommendation"],
    )
    natal_df = load_csv(
        NATAL_INPUT_PATH,
        ["regime", "component_name", "relative_regime_strength", "confidence_score", "dynamic_weight_recommendation"],
    )

    if not SUMMARY_INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing required file: {SUMMARY_INPUT_PATH}")

    v4_config = dict(base_config)
    v4_config["model_version"] = "astro_v4_regime_aware"
    v4_config["base_config_path"] = str(BASE_CONFIG_PATH)
    v4_config["dynamic_weight_profiles"] = build_dynamic_profiles(planet_df, aspect_df, natal_df)
    v4_config["market_regime_detector"] = {
        "trend_sma_window_days": 200,
        "trend_distance_threshold": 0.05,
        "trend_return_window_days": 30,
        "trend_return_threshold": 0.05,
        "bull_label": "Bull Market",
        "bear_label": "Bear Market",
        "sideways_label": "Sideways",
    }
    v4_config["volatility_state_detector"] = {
        "volatility_window_days": 30,
        "low_quantile": 0.30,
        "high_quantile": 0.70,
        "low_label": "Low Volatility",
        "high_label": "High Volatility",
        "midpoint_resolution": "nearest_anchor",
    }
    v4_config["analysis_inputs"] = {
        "regime_planet_importance": str(PLANET_INPUT_PATH),
        "regime_aspect_importance": str(ASPECT_INPUT_PATH),
        "regime_natal_importance": str(NATAL_INPUT_PATH),
        "regime_summary": str(SUMMARY_INPUT_PATH),
    }

    OUTPUT_PATH.write_text(json.dumps(v4_config, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
