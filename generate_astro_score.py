from datetime import datetime, timedelta, timezone
import pandas as pd
import swisseph as swe
import os
import requests

# -------------------------
# CONFIG
# -------------------------
NATAL_DT = datetime(2009, 1, 12, 3, 30, 25, tzinfo=timezone.utc)
LAT = 34.1070
LON = -118.0570

START_DATE = datetime(2009, 1, 12, tzinfo=timezone.utc)
END_DATE = datetime(2030, 12, 31, tzinfo=timezone.utc)

swe.set_sid_mode(swe.SIDM_LAHIRI)

PLANETS = {
    "Sun": swe.SUN,
    "Moon": swe.MOON,
    "Mercury": swe.MERCURY,
    "Venus": swe.VENUS,
    "Mars": swe.MARS,
    "Jupiter": swe.JUPITER,
    "Saturn": swe.SATURN,
    "Uranus": swe.URANUS,
    "Neptune": swe.NEPTUNE,
    "Pluto": swe.PLUTO,
}

PLANET_WEIGHTS = {
    "Jupiter": 2.0,
    "Saturn": 2.0,
    "Pluto": 2.0,
    "Uranus": 1.5,
    "Neptune": 1.2,
    "Mars": 1.0,
    "Mercury": 0.7,
    "Moon": 0.5,
    "Venus": 0.4,
    "Sun": 0.0,
}

ASPECTS = {
    "conjunction": (0, 1.00),
    "opposition": (180, 0.90),
    "square": (90, 0.85),
    "trine": (120, 0.75),
    "sextile": (60, 0.55),
}

TARGET_WEIGHTS = {
    "Asc": 1.20,
    "MC": 1.20,
    "Sun": 1.00,
    "Moon": 1.00,
    "Jupiter": 1.10,
    "Saturn": 1.10,
}

MAX_ORB_BY_PLANET = {
    "Jupiter": 3.0,
    "Saturn": 3.0,
    "Uranus": 3.0,
    "Neptune": 3.0,
    "Pluto": 3.0,
    "Mars": 2.0,
    "Mercury": 1.5,
    "Venus": 1.5,
    "Moon": 1.0,
    "Sun": 1.5,
}


def julday(dt):
    return swe.julday(dt.year, dt.month, dt.day, dt.hour + dt.minute / 60 + dt.second / 3600)


def norm360(x):
    return x % 360


def angle_diff(a, b):
    d = abs(norm360(a) - norm360(b))
    return min(d, 360 - d)


def get_planet_lon(jd, planet_id):
    xx, _ = swe.calc_ut(jd, planet_id, swe.FLG_SIDEREAL)
    return xx[0]


def get_houses(jd, lat, lon):
    cusps, ascmc = swe.houses_ex(jd, lat, lon, b'P', swe.FLG_SIDEREAL)
    return cusps, ascmc


def classify_regime(score):
    if score >= 3.0:
        return "strong_bull"
    elif score >= 1.5:
        return "bull"
    elif score > -1.5:
        return "neutral"
    elif score > -3.0:
        return "bear"
    return "crash_risk"


def aspect_score(transit_planet_name, transit_lon, target_name, target_lon):
    base_planet_weight = PLANET_WEIGHTS.get(transit_planet_name, 0.0)
    max_orb = MAX_ORB_BY_PLANET.get(transit_planet_name, 2.0)
    target_weight = TARGET_WEIGHTS.get(target_name, 1.0)

    best = 0.0
    for _, (aspect_angle, aspect_weight) in ASPECTS.items():
        d = angle_diff(transit_lon, target_lon)
        orb = abs(d - aspect_angle)
        if orb <= max_orb:
            orb_factor = max(0.0, 1 - orb / max_orb)
            score = base_planet_weight * aspect_weight * target_weight * orb_factor
            if score > best:
                best = score
    return best


def fetch_btc_price_history():
    """
    ใช้ CoinGecko market_chart/range แบบแบ่งเป็นช่วง ๆ
    เพราะทำงานได้เสถียรกว่า days=max ในหลาย environment
    """
    start_ts = int(datetime(2009, 1, 12, tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime.now(timezone.utc).timestamp())

    # แบ่งเป็น chunk ละ 365 วัน
    chunk_seconds = 365 * 24 * 60 * 60
    all_rows = []

    current_from = start_ts
    while current_from < end_ts:
        current_to = min(current_from + chunk_seconds, end_ts)

        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
        params = {
            "vs_currency": "usd",
            "from": current_from,
            "to": current_to,
        }

        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        if "prices" not in data:
            raise ValueError(f"CoinGecko response missing 'prices': {data}")

        for item in data["prices"]:
            ts_ms, price = item
            all_rows.append((ts_ms, price))

        current_from = current_to + 1

    df = pd.DataFrame(all_rows, columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
    df = df.groupby("date", as_index=False)["price"].last()
    return df


# Natal chart
natal_jd = julday(NATAL_DT)
_, natal_ascmc = get_houses(natal_jd, LAT, LON)

natal = {}
for name, pid in PLANETS.items():
    natal[name] = get_planet_lon(natal_jd, pid)
natal["Asc"] = natal_ascmc[0]
natal["MC"] = natal_ascmc[1]

rows = []
cur = START_DATE

while cur <= END_DATE:
    jd = julday(cur)
    transit_positions = {name: get_planet_lon(jd, pid) for name, pid in PLANETS.items()}

    expansion = 0.0
    contraction = 0.0
    narrative = 0.0
    trigger = 0.0

    for tname, tlon in transit_positions.items():
        for target in ["Sun", "Moon", "Jupiter", "Saturn", "Asc", "MC"]:
            score = aspect_score(tname, tlon, target, natal[target])

            if tname in ["Jupiter", "Uranus"]:
                expansion += score
            elif tname == "Saturn":
                contraction += score
            elif tname == "Pluto":
                expansion += score * 0.6
                contraction += score * 0.4
            elif tname == "Neptune":
                narrative += score
            elif tname in ["Mars", "Mercury", "Moon"]:
                trigger += score

    astro_momentum = expansion - contraction + 0.5 * narrative + 0.5 * trigger
    regime = classify_regime(astro_momentum)

    rows.append({
        "date": cur.date().isoformat(),
        "expansion_score": round(expansion, 4),
        "contraction_score": round(contraction, 4),
        "narrative_score": round(narrative, 4),
        "trigger_score": round(trigger, 4),
        "astro_momentum": round(astro_momentum, 4),
        "regime": regime,
    })

    cur += timedelta(days=1)

astro_df = pd.DataFrame(rows)
astro_df["date"] = pd.to_datetime(astro_df["date"])

# ดึงราคามารวม
price_df = fetch_btc_price_history()

# merge
df = astro_df.merge(price_df, on="date", how="left")

os.makedirs("data", exist_ok=True)
df.to_csv("data/bitcoin_astro_daily_score.csv", index=False)
print("Saved: data/bitcoin_astro_daily_score.csv")
