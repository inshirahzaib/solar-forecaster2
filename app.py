"""
Solar Forecast Skill Lab
Does a machine-learning solar forecast actually beat the simplest guesses?
Run with:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import forecasting as fc

st.set_page_config(page_title="Solar Forecast Skill Lab", page_icon="☀️", layout="wide")

BLUE, ORANGE, GREY, GREEN = "#2f6fdd", "#f28e2b", "#8a8f98", "#3aa655"


# ----------------------------------------------------------------------------
# cached helpers
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_demo():
    return fc.generate_demo_data()


@st.cache_data(show_spinner=False)
def cached_regular(df, time_col, target, extras):
    return fc.to_regular_series(df, time_col, target, list(extras))


@st.cache_data(show_spinner=False)
def cached_experiment(df, target, extras, step_h, horizon_h, test_frac, model_name, variable_pct):
    return fc.run_experiment(df, target, list(extras), step_h, horizon_h, test_frac,
                             model_name, variable_pct)


def fmt(x, digits=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{digits}f}"


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.0f}%"


# ----------------------------------------------------------------------------
# sidebar: choose data + settings
# ----------------------------------------------------------------------------
st.sidebar.title("☀️ Settings")
source = st.sidebar.radio(
    "1. Where does the data come from?",
    ["Demo data (synthetic, northern)", "Real data from NASA POWER", "Upload my own CSV"],
)

raw, source_note = None, ""

if source.startswith("Demo"):
    raw = cached_demo()
    source_note = ("**Demo data is synthetic** (computer-generated to look like Whitehorse, Yukon: "
                   "real sun geometry + random weather). Great for learning how the tool works, "
                   "but do not quote its results as real findings. Switch to NASA POWER or upload "
                   "real data for that.")

elif source.startswith("Real"):
    place = st.sidebar.selectbox("Location", list(fc.NASA_PLACES) + ["Custom lat/lon"])
    if place == "Custom lat/lon":
        lat = st.sidebar.number_input("Latitude", value=60.72, format="%.4f")
        lon = st.sidebar.number_input("Longitude", value=-135.06, format="%.4f")
    else:
        lat, lon = fc.NASA_PLACES[place]
    c1, c2 = st.sidebar.columns(2)
    y0 = c1.number_input("From year", 2010, 2026, 2022)
    y1 = c2.number_input("To year", 2010, 2026, 2023)
    if st.sidebar.button("⬇️ Download from NASA POWER"):
        if y1 < y0 or y1 - y0 > 3:
            st.sidebar.error("Pick 1 to 4 years (From year <= To year).")
        else:
            try:
                with st.spinner("Asking NASA for hourly data... (about 10-30 seconds per year)"):
                    st.session_state["nasa"] = (f"{place} ({lat:.2f}, {lon:.2f}), {y0}-{y1}",
                                                fc.fetch_nasa_power(lat, lon, y0, y1))
            except Exception as e:  # noqa: BLE001
                st.sidebar.error(f"NASA download failed: {e}. You can download the CSV by hand "
                                 "(see README) and use 'Upload my own CSV'.")
    if "nasa" in st.session_state:
        label, raw = st.session_state["nasa"]
        source_note = f"**Real NASA POWER data:** {label}. Sunlight is in W/m² (all-sky, on a flat surface)."

else:
    up = st.sidebar.file_uploader("Upload a CSV (hourly or finer)", type=["csv", "txt"])
    st.sidebar.caption("Needs a time column and a number column (sunlight or solar power). "
                       "NASA POWER CSV downloads work as they are.")
    if up is not None:
        try:
            raw = fc.read_csv_bytes(up.getvalue())
            source_note = f"**Your file:** {up.name}"
        except Exception as e:  # noqa: BLE001
            st.sidebar.error(f"Could not read that file: {e}")

st.sidebar.markdown("---")

# ----------------------------------------------------------------------------
# main page header
# ----------------------------------------------------------------------------
st.title("☀️ Solar Forecast Skill Lab")
st.caption("Does a machine-learning solar forecast actually beat the simplest guesses, "
           "and does that advantage shrink when the sky gets jumpy? "
           "Built for a TKS focus on AI for solar energy and northern microgrids.")

if raw is None:
    st.info("👈 Pick a data source in the sidebar. For NASA POWER, press the download button. "
            "For your own file, upload it.")
    st.stop()

st.markdown(source_note)

# column selection
time_default = fc.guess_time_column(raw)
cols = list(raw.columns)
with st.sidebar.expander("2. Columns (auto-detected)", expanded=False):
    time_col = st.selectbox("Time column", cols, index=cols.index(time_default))
    num_cols = fc.numeric_columns(raw, exclude=(time_col,))
    if not num_cols:
        st.error("No numeric columns found.")
        st.stop()
    tgt_default = fc.guess_target_column(raw, time_col)
    target = st.selectbox("What to forecast", num_cols, index=num_cols.index(tgt_default))
    extras = st.multiselect("Extra inputs (e.g. temperature)",
                            [c for c in num_cols if c != target], default=[])

st.sidebar.subheader("3. Forecast setup")
horizon_h = st.sidebar.select_slider("How far ahead (hours)", options=[1, 2, 3, 6, 12, 24, 48], value=1,
                                     help="1 hour is the standard solar test. Longer horizons make 'persistence' look unfairly bad.")
test_pct = st.sidebar.slider("Share of data held back for testing (%)", 10, 50, 25,
                             help="Always the most recent chunk of time, never shuffled.")
model_name = st.sidebar.selectbox("Model", fc.MODEL_CHOICES)
variable_pct = st.sidebar.slider("'Variable day' cut-off (percentile)", 30, 80, 50,
                                 help="Days jumpier than this percentile count as 'variable'. 50 = the jumpiest half.")

# prepare + run
try:
    df, step_h = cached_regular(raw, time_col, target, tuple(extras))
except Exception as e:  # noqa: BLE001
    st.error(f"Could not prepare your data: {e}")
    st.stop()

try:
    with st.spinner("Training the model and scoring everything..."):
        res = cached_experiment(df, target, tuple(extras), step_h, horizon_h,
                                test_pct / 100, model_name, variable_pct)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not run the forecast: {e}")
    st.stop()

test, groups = res["test"], res["groups"]
g = groups.set_index("group")

with st.expander("Preview of your data"):
    st.write(f"{len(df):,} rows - from {df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d} - "
             f"one row every {step_h:g} h - {int(df[target].isna().sum())} missing values after cleaning.")
    st.dataframe(df.head(48))

tab_res, tab_charts, tab_check, tab_gloss = st.tabs(
    ["📊 Results", "📈 Charts", "🔍 Checks & hand-check sheet", "📖 What everything means"])

# ----------------------------------------------------------------------------
# TAB 1: results
# ----------------------------------------------------------------------------
with tab_res:
    st.subheader(f"Results: {model_name.split(' (')[0]}, forecasting {horizon_h} hour(s) ahead")
    st.caption("Everything below is scored on the held-back TEST period, data the model never saw. "
               "Headline numbers use DAYTIME rows only, because night is always zero and makes every forecast look good.")

    for note in fc.honesty_checks(res, horizon_h):
        st.warning(note)

    d = g.loc["Daytime only"] if "Daytime only" in g.index else groups.iloc[0]
    st.info("**Finding:** " + fc.plain_english_finding(groups))

    c = st.columns(5)
    c[0].metric("Model RMSE", fmt(d["rmse_model"]), help="Typical size of the model's miss (big misses count extra). Lower is better.")
    c[1].metric("Persistence RMSE", fmt(d["rmse_persistence"]), help="Same measure for the 'next hour = this hour' guess.")
    c[2].metric("Yesterday RMSE", fmt(d["rmse_yesterday"]), help="Same measure for the 'same time as yesterday' guess.")
    c[3].metric("Skill vs persistence", pct(d["skill_vs_persistence"]), help="1 - model error / baseline error. 0% = no better than the baseline. Negative = worse.")
    c[4].metric("Skill vs yesterday", pct(d["skill_vs_yesterday"]), help="Same, against the tougher daily-pattern baseline.")

    c = st.columns(5)
    c[0].metric("Model MAE", fmt(d["mae_model"]), help="Plain average miss.")
    c[1].metric("Bias", fmt(d["bias"]), help="Average (forecast - actual). Positive = tends to over-forecast.")
    c[2].metric("R²", fmt(d["r2"], 3), help="Share of the ups and downs the model explains. 1 = perfect, 0 = no better than always guessing the average.")
    c[3].metric("Worst over-forecast", fmt(res["worst_over"]), help="Biggest single time the model promised more sun than arrived (daytime). This is the 'surprise shortfall' a battery or diesel must cover.")
    c[4].metric("95th-pct over-forecast", fmt(res["p95_over"]), help="On 1 daytime hour in 20 the model over-forecast by at least this much.")
    st.caption(f"Units: same units as '{target}'. Skill is based on RMSE.")

    st.markdown("#### Where does the model's advantage survive, and where does it shrink?")
    show = groups.rename(columns={
        "group": "Group", "rows": "Rows",
        "rmse_persistence": "Persistence RMSE", "rmse_yesterday": "Yesterday RMSE", "rmse_model": "Model RMSE",
        "mae_model": "Model MAE", "bias": "Bias", "r2": "R²",
    }).copy()
    show["Skill vs persistence"] = (groups["skill_vs_persistence"] * 100).round(0).astype("Int64").astype(str) + "%"
    show["Skill vs yesterday"] = (groups["skill_vs_yesterday"] * 100).round(0).astype("Int64").astype(str) + "%"
    show = show[["Group", "Rows", "Persistence RMSE", "Yesterday RMSE", "Model RMSE",
                 "Skill vs persistence", "Skill vs yesterday", "Model MAE", "Bias", "R²"]]
    st.dataframe(show.round(3), hide_index=True)
    st.download_button("Download this table (CSV)", groups.to_csv(index=False).encode(), "skill_table.csv", "text/csv")

# ----------------------------------------------------------------------------
# TAB 2: charts
# ----------------------------------------------------------------------------
with tab_charts:
    # --- chart 1: forecast vs actual
    st.markdown("#### 1. Forecast vs actual (zoom in on a few days)")
    tmin, tmax = test["target_time"].min().date(), test["target_time"].max().date()
    cc = st.columns(2)
    start = cc[0].date_input("Start day", value=tmin, min_value=tmin, max_value=tmax)
    ndays = cc[1].slider("Number of days", 1, 14, 4)
    w = test[(test["target_time"] >= pd.Timestamp(start)) &
             (test["target_time"] < pd.Timestamp(start) + pd.Timedelta(days=ndays))]
    fig = go.Figure()
    fig.add_scatter(x=w["target_time"], y=w["y_true"], name="Actual", line=dict(color="#222222", width=3))
    fig.add_scatter(x=w["target_time"], y=w["model"], name="Model", line=dict(color=BLUE, width=2))
    fig.add_scatter(x=w["target_time"], y=w["persistence"], name="Persistence", line=dict(color=ORANGE, width=1.5, dash="dot"))
    fig.add_scatter(x=w["target_time"], y=w["yesterday"], name="Same time yesterday", line=dict(color=GREEN, width=1.5, dash="dash"))
    fig.update_layout(height=420, yaxis_title=target, xaxis_title="Time being forecast", legend=dict(orientation="h"))
    st.plotly_chart(fig)
    st.caption("Persistence is the actual curve slid to the right (it always lags). Yesterday's line copies yesterday's weather, so it is wrong whenever today's weather differs.")

    # --- chart 2: RMSE by group
    sub = groups[groups["group"] != "All hours (including night)"]
    st.markdown("#### 2. Typical error by group (lower is better)")
    fig = go.Figure()
    fig.add_bar(x=sub["group"], y=sub["rmse_persistence"], name="Persistence", marker_color=ORANGE)
    fig.add_bar(x=sub["group"], y=sub["rmse_yesterday"], name="Same time yesterday", marker_color=GREEN)
    fig.add_bar(x=sub["group"], y=sub["rmse_model"], name="Model", marker_color=BLUE)
    fig.update_layout(barmode="group", height=430, yaxis_title=f"RMSE ({target})", legend=dict(orientation="h"))
    st.plotly_chart(fig)

    # --- chart 3: skill by group
    st.markdown("#### 3. Skill score by group (higher is better, below 0 = worse than the baseline)")
    fig = go.Figure()
    fig.add_bar(x=sub["group"], y=sub["skill_vs_persistence"] * 100, name="Skill vs persistence", marker_color=ORANGE)
    fig.add_bar(x=sub["group"], y=sub["skill_vs_yesterday"] * 100, name="Skill vs yesterday", marker_color=GREEN)
    fig.add_hline(y=0, line_color="black")
    fig.update_layout(barmode="group", height=430, yaxis_title="Skill (%)", legend=dict(orientation="h"))
    st.plotly_chart(fig)

    # --- chart 4: error by hour
    st.markdown("#### 4. Average miss by time of day")
    dd = test[test["is_day"]].copy()
    dd["hour"] = pd.DatetimeIndex(dd["target_time"]).hour
    by_h = pd.DataFrame({
        "Model": (dd["model"] - dd["y_true"]).abs().groupby(dd["hour"]).mean(),
        "Persistence": (dd["persistence"] - dd["y_true"]).abs().groupby(dd["hour"]).mean(),
        "Same time yesterday": (dd["yesterday"] - dd["y_true"]).abs().groupby(dd["hour"]).mean(),
    })
    fig = go.Figure()
    for name, colr in (("Persistence", ORANGE), ("Same time yesterday", GREEN), ("Model", BLUE)):
        fig.add_scatter(x=by_h.index, y=by_h[name], name=name, line=dict(color=colr))
    fig.update_layout(height=380, xaxis_title="Hour of day being forecast", yaxis_title=f"MAE ({target})", legend=dict(orientation="h"))
    st.plotly_chart(fig)

    # --- chart 5: scatter
    st.markdown("#### 5. Predicted vs actual (every dot is one daytime hour)")
    sc = dd.sample(min(3000, len(dd)), random_state=0) if len(dd) else dd
    mx = float(max(sc["y_true"].max(), sc["model"].max())) if len(sc) else 1.0
    fig = go.Figure()
    fig.add_scatter(x=sc["y_true"], y=sc["model"], mode="markers", name="Model",
                    marker=dict(color=BLUE, size=5, opacity=0.35))
    fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", name="Perfect forecast", line=dict(color="black", dash="dash"))
    fig.update_layout(height=450, xaxis_title=f"Actual ({target})", yaxis_title=f"Model forecast ({target})", legend=dict(orientation="h"))
    st.plotly_chart(fig)

    # --- chart 6: importance
    st.markdown("#### 6. Which inputs did the model rely on?")
    imp = res["importance"].head(10).iloc[::-1]
    fig = go.Figure(go.Bar(x=imp["error_increase"], y=imp["input"], orientation="h", marker_color=BLUE))
    fig.update_layout(height=380, xaxis_title=f"How much the error grows if this input is scrambled ({target} units)")
    st.plotly_chart(fig)

# ----------------------------------------------------------------------------
# TAB 3: checks + spreadsheet hand-check
# ----------------------------------------------------------------------------
with tab_check:
    st.markdown("#### How this test was set up")
    tr, te = res["train_range"], res["test_range"]
    st.markdown(
        f"- **Training period:** {tr[0]:%Y-%m-%d} to {tr[1]:%Y-%m-%d} ({res['n_train']:,} forecasts), seasons seen: {', '.join(res['train_seasons'])}\n"
        f"- **Test period:** {te[0]:%Y-%m-%d} to {te[1]:%Y-%m-%d} ({res['n_test']:,} forecasts), seasons: {', '.join(res['test_seasons'])}\n"
        f"- **Purge gap:** training rows whose answer falls inside the test period were removed (prevents leakage).\n"
        f"- **Steady vs variable:** {res['n_days_classified']} days scored for jumpiness; a day counts as variable above a score of {res['variable_threshold']:.3f} (the {variable_pct}th percentile).\n"
        f"- **Inputs given to the model ({len(res['feature_cols'])}):** {', '.join(res['feature_cols'])}")

    st.markdown("#### Biggest surprise (microgrid view)")
    if res["worst_over_time"] is not None:
        st.write(f"The worst daytime over-forecast was **{fmt(res['worst_over'])} {target}** at "
                 f"**{res['worst_over_time']:%Y-%m-%d %H:%M}** (time the forecast was made). "
                 "A microgrid relying on this forecast would have needed that much battery or diesel as backup.")

    st.markdown("#### Hand-check sheet (for your Google Sheets practice)")
    st.caption("Pick a week, download it, and rebuild the MAE / RMSE / skill score yourself in a spreadsheet. "
               "Your numbers should match the ones shown below. That match is your proof you understand the metrics.")
    wk = st.date_input("Week starts on", value=tmin, min_value=tmin, max_value=tmax, key="wk")
    week = test[(test["target_time"] >= pd.Timestamp(wk)) &
                (test["target_time"] < pd.Timestamp(wk) + pd.Timedelta(days=7))]
    if len(week) < 24:
        st.info("Not enough rows in that week. Choose another start date.")
    else:
        sheet = pd.DataFrame({
            "time": week["target_time"].dt.strftime("%Y-%m-%d %H:%M"),
            "actual": week["y_true"].round(3),
            "persistence_forecast": week["persistence"].round(3),
            "yesterday_forecast": week["yesterday"].round(3),
            "model_forecast": week["model"].round(3),
        })
        st.download_button("⬇️ Download week as CSV", sheet.to_csv(index=False).encode(),
                           "hand_check_week.csv", "text/csv")
        s = fc.score_rows(week)
        st.table(pd.DataFrame({
            "": ["Persistence", "Same time yesterday", "Model"],
            "MAE": [round(s["mae_persistence"], 3), round(s["mae_yesterday"], 3), round(s["mae_model"], 3)],
            "RMSE": [round(s["rmse_persistence"], 3), round(s["rmse_yesterday"], 3), round(s["rmse_model"], 3)],
        }).set_index(""))
        st.write(f"Rows in this week: **{s['rows']}** (all hours, night included). "
                 f"Skill vs persistence (RMSE): **{pct(s['skill_vs_persistence'])}**; "
                 f"vs yesterday: **{pct(s['skill_vs_yesterday'])}**.")

# ----------------------------------------------------------------------------
# TAB 4: glossary
# ----------------------------------------------------------------------------
with tab_gloss:
    st.markdown("""
### The forecasters being compared
- **Persistence:** "the future looks like right now." To forecast 1 hour ahead, copy the latest value.
- **Same time yesterday:** copy the value from the same clock time yesterday. A strong cheap baseline, because the sun follows a daily cycle.
- **Model:** a machine-learning model trained on past data. It is given recent values, trends, time of day, and time of year.

### Error numbers (all in the same units as your data)
- **Error:** forecast minus actual. Forecast 300, actual 350 -> error -50.
- **MAE (mean absolute error):** ignore plus/minus, average the size of the misses. "On average we are off by this much."
- **RMSE (root mean square error):** square each error, average, square-root. Big misses count extra, so RMSE >= MAE. If RMSE is much bigger than MAE, there are a few large misses (usually cloud surprises).
- **Bias:** the average signed error. Positive means it tends to promise too much sun.
- **R²:** how much of the ups and downs the model explains (1 perfect, 0 = no better than guessing the average).
- **Skill score:** 1 - (model error / baseline error). 0% = no better than the baseline, 100% = perfect, **negative = worse than the baseline**. This is the number forecasters use to decide if a model is worth deploying.

### Groups in the table
- **All hours vs Daytime only:** night is always 0, which every forecaster gets right, so it flatters all of them. Daytime-only is the honest view.
- **Steady vs Variable days:** each day gets a "jumpiness" score from how much the sunlight ratio (actual / normal for that time) jumps from hour to hour. Clear AND fully overcast days are steady. Days with moving clouds are variable.
- **Seasons:** Winter = Dec-Feb, Spring = Mar-May, Summer = Jun-Aug, Autumn = Sep-Nov (northern hemisphere).

### Setup words
- **Horizon:** how far ahead you forecast.
- **Train / test split:** the model learns on the past and is graded on a later period it has never seen. Never shuffled, because real forecasts always predict the future.
- **Leakage:** accidentally letting the model see information from the future. This tool blocks it with a purge gap.
- **Permutation importance (chart 6):** scramble one input and see how much worse the model gets. Bigger = the model leaned on it more.
- **Units:** NASA POWER sunlight is W/m² (watts per square metre of flat surface). Solar power data is usually kW.

### How to read the main result
A model's advantage over persistence is almost always smaller on variable days, and that is exactly when a microgrid needs the forecast most. The question to ask of any published "accuracy" number is: *compared with what, and on which kind of day?*
""")
