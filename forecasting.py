"""
forecasting.py - all the "thinking" code for the Solar Forecast Skill Lab.

app.py only draws the website. Everything that loads data, builds baselines,
trains the model and scores it lives here, so you can read it top to bottom.
"""
import io

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEASONS = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]

MODEL_CHOICES = [
    "Gradient boosting (HistGradientBoosting)",
    "Linear regression (Ridge) - simple comparison model",
]

NASA_PLACES = {
    "Whitehorse, Yukon": (60.7211, -135.0568),
    "Yellowknife, NWT": (62.4540, -114.3718),
    "Inuvik, NWT": (68.3607, -133.7230),
    "Iqaluit, Nunavut": (63.7467, -68.5170),
    "Toronto, Ontario (southern comparison)": (43.6532, -79.3832),
}


# ----------------------------------------------------------------------------
# 1. GETTING DATA
# ----------------------------------------------------------------------------
def generate_demo_data(lat=60.72, seed=42):
    """
    Make 2 years of SYNTHETIC hourly solar data for a Whitehorse-like site.

    Real physics for the sun's height in the sky (so winter days are short and
    weak, summer days are long) + a random "weather" layer (clear days, broken
    cloud days, overcast days). It is realistic in SHAPE but it is not real
    measurements - never quote findings from it as if they were real.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", "2023-12-31 23:00", freq="h")
    doy = idx.dayofyear.to_numpy()
    hour = idx.hour.to_numpy() + 0.5

    decl = np.radians(23.44) * np.sin(2 * np.pi * (doy - 81) / 365)
    ha = np.radians(15 * (hour - 12))
    lat_r = np.radians(lat)
    sin_elev = np.sin(lat_r) * np.sin(decl) + np.cos(lat_r) * np.cos(decl) * np.cos(ha)
    sin_elev = np.clip(sin_elev, 0, None)
    clear_sky = 1050 * sin_elev ** 1.15  # W/m2 if the sky were perfectly clear

    # One weather "mood" per day: 0 = clear, 1 = broken cloud (variable), 2 = overcast
    day_ids = np.asarray(idx.normalize().factorize()[0])
    n_days = day_ids.max() + 1
    base_kt, vol = np.zeros(n_days), np.zeros(n_days)
    kt_mean = [0.80, 0.55, 0.25]
    kt_vol = [0.03, 0.22, 0.06]
    state = 1
    for d in range(n_days):
        if d == 0 or rng.random() < 0.4:
            state = rng.choice(3, p=[0.30, 0.40, 0.30])
        base_kt[d] = kt_mean[state] + rng.normal(0, 0.04)
        vol[d] = kt_vol[state]

    noise = np.zeros(len(idx))
    eps = rng.normal(0, 1, len(idx))
    for t in range(1, len(idx)):
        noise[t] = 0.65 * noise[t - 1] + vol[day_ids[t]] * eps[t]
    clearness = np.clip(base_kt[day_ids] + noise, 0.03, 0.85)

    ghi = clear_sky * clearness
    ghi[sin_elev <= 0] = 0.0

    temp = (-1.5 + 15.5 * np.sin(2 * np.pi * (doy - 110) / 365)
            + 3 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 2, len(idx)))

    return pd.DataFrame({"timestamp": idx, "ghi_wm2": np.round(ghi, 1),
                         "temp_c": np.round(temp, 1)})


def fetch_nasa_power(lat, lon, start_year, end_year):
    """Download hourly sunlight + temperature from NASA POWER (free, no key)."""
    import requests

    frames = []
    for year in range(int(start_year), int(end_year) + 1):
        resp = requests.get(
            "https://power.larc.nasa.gov/api/temporal/hourly/point",
            params={
                "start": f"{year}0101", "end": f"{year}1231",
                "latitude": lat, "longitude": lon, "community": "RE",
                "parameters": "ALLSKY_SFC_SW_DWN,T2M",
                "format": "JSON", "time-standard": "LST",
            },
            timeout=180,
        )
        resp.raise_for_status()
        block = pd.DataFrame(resp.json()["properties"]["parameter"])
        block.index = pd.to_datetime(block.index, format="%Y%m%d%H")
        frames.append(block)
    out = pd.concat(frames).sort_index().replace(-999, np.nan)
    out.index.name = "timestamp"
    out = out.rename(columns={"ALLSKY_SFC_SW_DWN": "ghi_wm2", "T2M": "temp_c"})
    return out.reset_index()


def read_csv_bytes(raw):
    """
    Read an uploaded CSV. Also copes with the format NASA POWER's website gives
    you (a block of header text, then YEAR,MO,DY,HR columns, and -999 for missing).
    """
    text = raw.decode("utf-8-sig", errors="replace")
    if "-END HEADER-" in text:
        text = text.split("-END HEADER-", 1)[1].lstrip("\r\n")
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.replace([-999, -999.0], np.nan)

    lower = {c.lower(): c for c in df.columns}
    if all(k in lower for k in ("year", "mo", "dy", "hr")):
        stamp = pd.to_datetime(pd.DataFrame({
            "year": df[lower["year"]], "month": df[lower["mo"]],
            "day": df[lower["dy"]], "hour": df[lower["hr"]],
        }))
        df = df.drop(columns=[lower[k] for k in ("year", "mo", "dy", "hr")])
        df.insert(0, "timestamp", stamp)
    return df


def guess_time_column(df):
    for c in df.columns:
        if any(k in c.lower() for k in ("time", "date", "stamp")):
            return c
    return df.columns[0]


def numeric_columns(df, exclude=()):
    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def guess_target_column(df, time_col):
    nums = numeric_columns(df, exclude=(time_col,))
    keys = ("ghi", "allsky", "sw_dwn", "irradiance", "power", "kw", "generation", "output", "solar")
    for c in nums:
        if any(k in c.lower() for k in keys):
            return c
    return nums[0] if nums else None


def to_regular_series(df, time_col, target, extras=()):
    """Clean the table into a perfectly regular time grid (one row per step)."""
    extras = [c for c in extras if c != target and c != time_col]
    d = df[[time_col, target] + extras].copy()

    t = pd.to_datetime(d[time_col], errors="coerce")
    try:
        if t.dt.tz is not None:
            t = t.dt.tz_localize(None)
    except (AttributeError, TypeError):
        t = pd.to_datetime(d[time_col], errors="coerce", utc=True).dt.tz_localize(None)
    d[time_col] = t
    d = d.dropna(subset=[time_col]).sort_values(time_col)
    d = d.drop_duplicates(subset=[time_col]).set_index(time_col)
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    if len(d) < 200:
        raise ValueError("Fewer than 200 usable rows. I need at least a few weeks of data.")
    step = d.index.to_series().diff().median()
    step_h = step.total_seconds() / 3600
    if step_h <= 0 or step_h > 3:
        raise ValueError(
            f"Your data steps are about {step_h:.1f} hours apart. I need hourly (or finer) "
            "data, because the daily sun cycle has to be visible.")
    d = d.resample(step).mean()
    d = d.interpolate(limit=3, limit_area="inside")  # patch gaps of up to 3 steps
    return d, step_h


# ----------------------------------------------------------------------------
# 2. BUILDING THE FORECASTING TABLE (features + baselines)
# ----------------------------------------------------------------------------
def build_dataset(df, target, extras, step_h, horizon_h):
    """
    Each row = "a moment when we make a forecast" (time t).
    y_true      = what actually happened `horizon` later (the answer).
    persistence = the last value we know (baseline 1).
    yesterday   = the value at the same clock time yesterday (baseline 2).
    All model inputs only use information available at time t (no peeking).
    """
    y = df[target]
    steps_per_day = int(round(24 / step_h))
    h = max(1, int(round(horizon_h / step_h)))
    days_back = int(np.ceil(h / steps_per_day))
    yday_shift = steps_per_day * days_back - h

    feats = pd.DataFrame(index=df.index)
    feats["now"] = y
    seen = set()
    for lag_h in (1, 2, 3, 6, 12, 24):
        k = int(round(lag_h / step_h))
        if k >= 1 and k not in seen:
            seen.add(k)
            feats[f"lag_{lag_h}h"] = y.shift(k)
    feats["same_time_yesterday"] = y.shift(yday_shift)
    for w_h in (3, 24):
        w = max(2, int(round(w_h / step_h)))
        feats[f"mean_{w_h}h"] = y.rolling(w).mean()
        feats[f"std_{w_h}h"] = y.rolling(w).std()

    target_time = df.index + pd.Timedelta(hours=h * step_h)
    hour_frac = target_time.hour + target_time.minute / 60
    feats["hour_sin_target"] = np.sin(2 * np.pi * hour_frac / 24)
    feats["hour_cos_target"] = np.cos(2 * np.pi * hour_frac / 24)
    feats["season_sin_target"] = np.sin(2 * np.pi * target_time.dayofyear / 365.25)
    feats["season_cos_target"] = np.cos(2 * np.pi * target_time.dayofyear / 365.25)
    for c in extras:
        if c != target and c in df.columns:
            feats[c] = df[c]

    out = feats.copy()
    out["y_true"] = y.shift(-h)
    out["persistence"] = y
    out["yesterday"] = y.shift(yday_shift)
    out["target_time"] = target_time
    feature_cols = list(feats.columns)
    out = out.dropna(subset=feature_cols + ["y_true", "persistence", "yesterday"])
    return out, feature_cols, h


def make_model(name):
    if name.startswith("Gradient"):
        return HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=30, l2_regularization=1.0, random_state=0)
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


# ----------------------------------------------------------------------------
# 3. DAYTIME + "STEADY vs VARIABLE DAY" LABELS
# ----------------------------------------------------------------------------
def climatology(y):
    """Average value for each (month, hour-of-day) - 'what is normal for this time'."""
    return y.groupby([y.index.month, y.index.hour]).transform("mean")


def daytime_mask(y, clim):
    """True when the sun is normally up at that month+hour (not decided by the outcome)."""
    return (clim > 0.01 * y.quantile(0.99)).fillna(False).astype(bool)


def day_variability(y, clim):
    """
    Score each calendar day by how jumpy the sunlight was.
    ratio = actual / normal-for-that-time. On a steady day (clear OR fully
    overcast) the ratio barely changes hour to hour. On a variable day it
    swings. Score = average absolute hour-to-hour change in that ratio,
    using only the core daylight hours of the day.
    """
    month_peak = clim.groupby(y.index.month).transform("max")
    core = (clim > 0.2 * month_peak) & (clim > 0)
    ratio = (y / clim.where(clim > 0)).where(core)
    change = ratio.diff().abs()
    dates = y.index.normalize()
    score = change.groupby(dates).mean()
    n_core = ratio.groupby(dates).count()
    return score.where(n_core >= 3)


# ----------------------------------------------------------------------------
# 4. SCORING
# ----------------------------------------------------------------------------
def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def skill(model_err, base_err):
    return float(1 - model_err / base_err) if base_err and base_err > 0 else np.nan


def score_rows(sub):
    """Score the model and both baselines on any subset of the test rows."""
    y = sub["y_true"].to_numpy()
    out = {"rows": int(len(sub))}
    for name, col in (("model", "model"), ("persistence", "persistence"), ("yesterday", "yesterday")):
        out[f"rmse_{name}"] = rmse(y, sub[col])
        out[f"mae_{name}"] = mae(y, sub[col])
    out["skill_vs_persistence"] = skill(out["rmse_model"], out["rmse_persistence"])
    out["skill_vs_yesterday"] = skill(out["rmse_model"], out["rmse_yesterday"])
    out["mae_skill_vs_persistence"] = skill(out["mae_model"], out["mae_persistence"])
    out["bias"] = float(np.mean(sub["model"] - sub["y_true"]))
    sst = float(np.sum((y - y.mean()) ** 2))
    out["r2"] = float(1 - np.sum((y - sub["model"]) ** 2) / sst) if sst > 0 else np.nan
    return out


def build_group_table(test, min_rows=24):
    day = test["is_day"].to_numpy()
    dtype = test["day_type"].to_numpy()
    season = test["season"].to_numpy()
    groups = [("All hours (including night)", np.ones(len(test), bool)),
              ("Daytime only", day)]
    for dt in ("Steady", "Variable"):
        groups.append((f"Daytime - {dt} days", day & (dtype == dt)))
    for s in SEASON_ORDER:
        m = day & (season == s)
        if m.sum() >= min_rows:
            groups.append((f"Daytime - {s}", m))
            for dt in ("Steady", "Variable"):
                m2 = m & (dtype == dt)
                if m2.sum() >= min_rows:
                    groups.append((f"Daytime - {s} - {dt} days", m2))
    rows = []
    for label, mask in groups:
        if mask.sum() >= min_rows:
            r = score_rows(test[mask])
            r["group"] = label
            rows.append(r)
    return pd.DataFrame(rows)


def run_experiment(df, target, extras, step_h, horizon_h, test_frac, model_name, variable_pct):
    """Train on the past, test on the future, score everything."""
    data, feature_cols, h = build_dataset(df, target, list(extras), step_h, horizon_h)
    if len(data) < 500:
        raise ValueError("Not enough clean rows after building lags (need 500+). Use more data.")

    split = int(len(data) * (1 - test_frac))
    test = data.iloc[split:].copy()
    # PURGE: drop training rows whose answer lies inside the test period (stops leakage)
    train = data[data["target_time"] < test.index[0]]
    if len(train) < 300:
        raise ValueError("Training set too small. Lower the test share or add more data.")

    model = make_model(model_name)
    model.fit(train[feature_cols], train["y_true"])
    pred = model.predict(test[feature_cols])
    floor = 0.0 if data["y_true"].min() >= 0 else -np.inf
    test["model"] = np.clip(pred, floor, None)
    test["error"] = test["model"] - test["y_true"]

    y = df[target]
    clim = climatology(y)
    day_mask = daytime_mask(y, clim)
    var_by_date = day_variability(y, clim)
    threshold = float(var_by_date.quantile(variable_pct / 100))

    tt = pd.DatetimeIndex(test["target_time"])
    test["is_day"] = day_mask.reindex(tt).fillna(False).to_numpy().astype(bool)
    test["date"] = tt.normalize()
    test["variability"] = test["date"].map(var_by_date)
    test["day_type"] = np.where(test["variability"].isna(), "Unclassified",
                                np.where(test["variability"] > threshold, "Variable", "Steady"))
    test["season"] = tt.month.map(SEASONS)

    groups = build_group_table(test)

    # Which inputs mattered? (shuffle one input at a time, see how much error grows)
    sample = test[test["is_day"]]
    sample = sample.sample(min(1500, len(sample)), random_state=0) if len(sample) else test.head(1500)
    imp = permutation_importance(model, sample[feature_cols], sample["y_true"], n_repeats=3,
                                 random_state=0, scoring="neg_root_mean_squared_error")
    importance = (pd.DataFrame({"input": feature_cols, "error_increase": imp.importances_mean})
                  .sort_values("error_increase", ascending=False).reset_index(drop=True))

    day_rows = test[test["is_day"]]
    over = day_rows["error"]
    return {
        "test": test, "groups": groups, "importance": importance,
        "feature_cols": feature_cols, "h_steps": h, "n_train": len(train), "n_test": len(test),
        "train_range": (train.index.min(), train.index.max()),
        "test_range": (test.index.min(), test.index.max()),
        "train_seasons": sorted(set(pd.DatetimeIndex(train["target_time"]).month.map(SEASONS))),
        "test_seasons": sorted(set(test["season"])),
        "variable_threshold": threshold,
        "n_days_classified": int(var_by_date.notna().sum()),
        "worst_over": float(over.max()) if len(over) else np.nan,
        "worst_over_time": over.idxmax() if len(over) else None,
        "p95_over": float(np.percentile(over, 95)) if len(over) else np.nan,
    }


# ----------------------------------------------------------------------------
# 5. PLAIN-ENGLISH CHECKS AND SUMMARY
# ----------------------------------------------------------------------------
def honesty_checks(res, horizon_h):
    notes = []
    t0, t1 = res["test_range"]
    days = (t1 - t0).days + 1
    if days < 60:
        notes.append(f"The test period is only {days} days long. Results from a short window can be luck. Use more data or a bigger test share.")
    missing = [s for s in res["test_seasons"] if s not in res["train_seasons"]]
    if missing:
        notes.append("Seasonal mismatch: the model is tested on " + ", ".join(missing) +
                     " but never saw that season in training. That is a hard (and realistic) test, "
                     "and it may make the model look worse than it would with a full year of training data.")
    if len(res["test_seasons"]) < 2:
        notes.append("The test period covers a single season, so any 'summer vs winter' comparison is not possible.")
    if horizon_h % 24 == 0:
        notes.append(f"At {horizon_h} hours ahead, 'persistence' and 'same time yesterday' are the same forecast, so the two skill scores are identical.")
    elif horizon_h >= 6:
        notes.append(f"At {horizon_h} hours ahead, plain persistence is a very weak baseline (the sun will have moved a lot). "
                     "Judge the model against 'same time yesterday' as well - it is the fairer test.")
    if res["n_days_classified"] < 30:
        notes.append("Fewer than 30 days could be classified as steady/variable, so that split is shaky.")
    return notes


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.0f}%"


def plain_english_finding(groups):
    g = groups.set_index("group")
    if "Daytime - Steady days" in g.index and "Daytime - Variable days" in g.index:
        s, v = g.loc["Daytime - Steady days"], g.loc["Daytime - Variable days"]
        return (f"Against plain persistence, the model's advantage was {pct(s['skill_vs_persistence'])} on steady days "
                f"but {pct(v['skill_vs_persistence'])} on variable days. Against 'same time yesterday' it was "
                f"{pct(s['skill_vs_yesterday'])} on steady days and {pct(v['skill_vs_yesterday'])} on variable days.")
    if "Daytime only" in g.index:
        d = g.loc["Daytime only"]
        return (f"In daytime hours the model beat persistence by {pct(d['skill_vs_persistence'])} "
                f"and 'same time yesterday' by {pct(d['skill_vs_yesterday'])}.")
    return "Not enough rows to summarise."
