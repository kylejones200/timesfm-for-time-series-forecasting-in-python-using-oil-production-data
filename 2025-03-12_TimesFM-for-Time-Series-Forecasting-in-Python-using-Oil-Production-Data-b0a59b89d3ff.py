# Description: Short example for TimesFM for Time Series Forecasting in Python using Oil Production Data.



# Load real oil production data

from data_io import read_csv
from dataclasses import dataclass
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timesfm

df = read_csv("north_dakota_production.csv")
# Select top two wells with nonzero oil production
top_wells = df[df["Oil"] > 0].nlargest(2, "Oil")["API_WELLNO"]
df = df[df["API_WELLNO"].isin(top_wells)].rename(columns={"API_WELLNO": "unique_id", "Date": "ds", "Oil": "y"})
df["ds"] = pd.to_datetime(df["ds"])

# Train-test split using TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5, test_size=int(0.2 * len(df)))
train_idx, test_idx = list(tscv.split(df))[-1]
train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

# Initialize TimesFM Model
tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        per_core_batch_size=32, horizon_len=128, input_patch_len=32, output_patch_len=128,
        num_layers=50, model_dims=1280, use_positional_embedding=False
    ),
    checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
)

# Generate forecast
forecast_df = tfm.forecast_on_df(inputs=train_df, freq="M", value_name="y", num_jobs=-1)
forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])
# Aggregate forecast
forecast_df = forecast_df.groupby("ds")["timesfm"].mean().reset_index()

# Restrict forecast to match test period
forecast_df = forecast_df[forecast_df["ds"].between(test_df["ds"].min(), test_df["ds"].max())]

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(df["ds"], df["y"], label="Monthly Oil Production",  color="black", alpha=0.3)
plt.plot(test_df["ds"], test_df["y"], label="Test Data", color="blue")
plt.plot(forecast_df["ds"], forecast_df["timesfm"], label="Forecast",  color="red")

# Save and show
plt.savefig("timesfm_test_forecast_tufte.png", dpi=300)
plt.show()


np.random.seed(42)
plt.rcParams.update({
    'axes.grid': False,'font.family': 'serif','axes.spines.top': False,'axes.spines.right': False,'axes.linewidth': 0.8})

def save_fig(path: str):
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()

@dataclass
class Config:
    csv_path: str = "2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    horizon: int = 8  # Jan-Aug 2025


def load_series(cfg: Config) -> pd.Series:
    p = Path(cfg.csv_path)
    df = read_csv(p, header=None, usecols=[0,1], names=["date","value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().sort_values("date").set_index("date")["value"].asfreq(cfg.freq)
    return s.astype(float)


def build_timesfm_model(h: int):
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu",
            per_core_batch_size=32,
            horizon_len=h,
            context_len=512,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
        ),
    )
    return tfm


def main():
    cfg = Config()
    y = load_series(cfg)

    # Train/cutoff at Dec 2024, forecast Jan-Aug 2025
    end_2024 = pd.Timestamp("2024-12-01")
    jan_2025 = pd.Timestamp("2025-01-01")
    aug_2025 = pd.Timestamp("2025-08-01")

    y_train = y.loc[:end_2024]
    y_act = y.loc[jan_2025:aug_2025]

    tfm = build_timesfm_model(h=len(pd.period_range('2025-01', '2025-08', freq='M')))

    # Prepare dataframe input for forecast_on_df
    df_in = pd.DataFrame({
        "unique_id": ["EIA"] * len(y_train),
        "ds": y_train.index,
        "y": y_train.values,
    })
    # TimesFM expects freq like "M" for monthly
    fc_df = tfm.forecast_on_df(inputs=df_in, freq="M", value_name="y", num_jobs=-1)
    # Filter and index
    fc_df = fc_df[fc_df["unique_id"] == "EIA"].copy()
    fc_df = fc_df.set_index("ds").sort_index()
    fc_df = fc_df.loc[jan_2025:aug_2025]
    # Determine forecast column name
    col = None
    for cand in ["y_hat", "yhat", "mean", "y", "point_forecast"]:
        if cand in fc_df.columns:
            col = cand; break
    if col is None:
        # pick first numeric column
        num_cols = [c for c in fc_df.columns if pd.api.types.is_numeric_dtype(fc_df[c])]
        if num_cols:
            col = num_cols[0]
        else:
            raise RuntimeError(f"No numeric forecast column found in TimesFM output: {list(fc_df.columns)}")

    # Plot greyscale Tufte-style
    start_2024 = pd.Timestamp("2024-01-01")
    y_hist = y.loc[start_2024:end_2024]

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(y_hist.index, y_hist.values, color="#888888", lw=1.5)
    ax.axvline(jan_2025, color="#666666", linestyle="--", lw=1)
    if len(y_act):
        ax.plot(y_act.index, y_act.values, color="#444444", lw=1.8)
    if not fc_df.empty:
        ax.plot(fc_df.index, fc_df[col].values, color="#000000", lw=2.0)

    from matplotlib.ticker import MaxNLocator, StrMethodFormatter
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('')

    if len(y_hist):
        ax.annotate('History (2024)', xy=(y_hist.index[-1], y_hist.values[-1]), xytext=(6,0), textcoords='offset points', fontsize=9, va='center', ha='left', color='#666666')
    if len(y_act):
        ax.annotate('Actual (Jan-Aug 2025)', xy=(y_act.index[-1], y_act.values[-1]), xytext=(6,0), textcoords='offset points', fontsize=9, va='center', ha='left', color='#444444')
    if not fc_df.empty:
        ax.annotate('TimesFM', xy=(fc_df.index[-1], fc_df[col].values[-1]), xytext=(6,0), textcoords='offset points', fontsize=9, va='center', ha='left', color='#000000')

    ax.set_title('EIA Net Generation — TimesFM forecast Jan-Aug 2025')
    save_fig('eia_timesfm_last_fold.png')

if __name__ == '__main__':
    main()
