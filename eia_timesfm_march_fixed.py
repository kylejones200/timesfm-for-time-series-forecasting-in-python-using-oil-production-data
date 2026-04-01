import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
import timesfm

np.random.seed(42)
plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
    }
)


def save_fig(path: str):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


@dataclass
class Config:
    csv_path: str = "2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    horizon: int = 8  # Jan–Aug 2025


def load_series(cfg: Config) -> pd.Series:
    p = Path(cfg.csv_path)
    df = pd.read_csv(p, header=None, usecols=[0, 1], names=["date", "value"], sep=",")
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

    # Train/cutoff at Dec 2024, forecast Jan–Aug 2025
    end_2024 = pd.Timestamp("2024-12-01")
    jan_2025 = pd.Timestamp("2025-01-01")
    aug_2025 = pd.Timestamp("2025-08-01")

    y_train = y.loc[:end_2024]
    y_act = y.loc[jan_2025:aug_2025]

    tfm = build_timesfm_model(h=len(pd.period_range("2025-01", "2025-08", freq="M")))

    # Prepare dataframe input for forecast_on_df
    df_in = pd.DataFrame(
        {
            "unique_id": ["EIA"] * len(y_train),
            "ds": y_train.index,
            "y": y_train.values,
        }
    )
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
            col = cand
            break
    if col is None:
        # pick first numeric column
        num_cols = [c for c in fc_df.columns if pd.api.types.is_numeric_dtype(fc_df[c])]
        if num_cols:
            col = num_cols[0]
        else:
            raise RuntimeError(
                f"No numeric forecast column found in TimesFM output: {list(fc_df.columns)}"
            )

    # Plot greyscale Tufte-style
    start_2024 = pd.Timestamp("2024-01-01")
    y_hist = y.loc[start_2024:end_2024]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_hist.index, y_hist.values, color="#888888", lw=1.5)
    ax.axvline(jan_2025, color="#666666", linestyle="--", lw=1)
    if len(y_act):
        ax.plot(y_act.index, y_act.values, color="#444444", lw=1.8)
    if not fc_df.empty:
        ax.plot(fc_df.index, fc_df[col].values, color="#000000", lw=2.0)

    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("")

    if len(y_hist):
        ax.annotate(
            "History (2024)",
            xy=(y_hist.index[-1], y_hist.values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            va="center",
            ha="left",
            color="#666666",
        )
    if len(y_act):
        ax.annotate(
            "Actual (Jan–Aug 2025)",
            xy=(y_act.index[-1], y_act.values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            va="center",
            ha="left",
            color="#444444",
        )
    if not fc_df.empty:
        ax.annotate(
            "TimesFM",
            xy=(fc_df.index[-1], fc_df[col].values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            va="center",
            ha="left",
            color="#000000",
        )

    ax.set_title("EIA Net Generation — TimesFM forecast Jan–Aug 2025")
    save_fig("eia_timesfm_last_fold.png")


if __name__ == "__main__":
    main()
