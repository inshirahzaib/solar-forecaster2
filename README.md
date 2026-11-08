# ☀️ Solar Forecast Skill Lab

Does a machine-learning solar forecast actually beat the simplest guesses, and does that
advantage shrink on jumpy (variable) days and in northern winters?

The app trains a model on the past, tests it on a later period it never saw, and scores it against
two baselines:

- **Persistence**: the future looks like right now.
- **Same time yesterday**: the daily pattern repeated.

It reports MAE, RMSE, R², bias and a **skill score**, split into daytime, steady days vs variable
days, and by season.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data options (sidebar)

1. **Demo data**: synthetic Whitehorse-like data built in. Needs nothing. Not real measurements.
2. **Real data from NASA POWER**: choose a place and years, press the download button.
3. **Upload my own CSV**: needs a time column and a numeric column.

### Downloading NASA POWER by hand (backup)

1. Go to https://power.larc.nasa.gov and open the **Data Access Viewer**.
2. Choose **Single Point**, **Hourly**, community **Renewable Energy**.
3. Enter the latitude and longitude (Whitehorse: 60.7211, -135.0568), then pick a date range.
4. Under parameters choose "All Sky Surface Shortwave Downward Irradiance" and "Temperature at 2 Meters".
5. Output format **CSV**, then Submit and download. Upload it with "Upload my own CSV".

## Files

| File | Purpose |
|---|---|
| `app.py` | The website (layout, charts) |
| `forecasting.py` | Data cleaning, baselines, model, scoring |
| `requirements.txt` | Python packages |
| `.devcontainer/devcontainer.json` | Auto-setup for GitHub Codespaces |

## Honest limitations

- One site and one model type, so this checks plausibility, not published papers' exact claims.
- Persistence is not the strongest baseline. Researchers often use "smart persistence" (clear-sky adjusted); that is a good next step.
- NASA POWER is satellite-derived estimates, not ground sensors.
