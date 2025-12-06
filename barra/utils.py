import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
import matplotlib.dates as mdates


def wide_weights_to_long(df: pd.DataFrame, date_col: str | None = None, normalize: bool = True) -> pd.DataFrame:
    if date_col is None:
        if df.index.name is not None:
            date_col = df.index.name
            df = df.reset_index()
        else:
            date_col = "date"
            df = df.rename(columns={df.columns[0]: date_col})
    else:
        if date_col not in df.columns:
            raise ValueError("date_col not found in wide weights dataframe")
    date = pd.to_datetime(df[date_col]).dt.date
    w = df.drop(columns=[date_col])
    long = w.set_index(date).stack(dropna=False).reset_index()
    long.columns = [date_col, "asset", "weight"]
    long["asset"] = long["asset"].astype(str).str.replace('.XSHE','.SZ',regex=False).str.replace('.XSHG','.SH',regex=False)
    if normalize:
        long["weight"] = long["weight"].astype(float)
        sums = long.groupby(date_col)["weight"].transform(lambda x: x.sum() if x.sum() != 0 else 1.0)
        long["weight"] = long["weight"] / sums
    return long


def plot_contribution_grid(series: dict, labels_order: list[str] | None = None, ncols: int = 2, figsize=(12, 16), out_path: str | None = None, start_from_one: bool = True):
    labels = labels_order or list(series.keys())
    if len(labels) == 0:
        raise ValueError("empty contribution series")
    n = len(labels)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for i, label in enumerate(labels):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]
        s = series[label].copy()
        p = s["portfolio"].astype(float).fillna(0)
        b = s["benchmark"].astype(float).fillna(0)
        a = s["active"].astype(float).fillna(0)
        if start_from_one:
            p = (1.0 + p).cumprod()
            b = (1.0 + b).cumprod()
            a = (1.0 + a).cumprod()
        else:
            p = p.cumsum(); b = b.cumsum(); a = a.cumsum()
        ax.plot(s.index, p, color="green", label="Portfolio")
        ax.plot(s.index, b, color="orange", label="Benchmark")
        ax.plot(s.index, a, color="steelblue", label="Active")
        ax.set_title(f"{label} Gross Return")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best", fontsize=8)
    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].axis("off")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140)
    return fig

def plot_factor_grid(series: dict, labels_order: list[str] | None = None, ncols: int = 2, figsize=(12, 16), out_path: str | None = None, start_from_one: bool = True, drop_last: bool = False):
    labels = labels_order or list(series.keys())
    if drop_last and len(labels) > 0:
        labels = labels[:-1]
    # 自动剔除可能的汇总条目
    labels = [l for l in labels if str(l).lower() not in {"active portfolio gross return", "汇总"}]
    if len(labels) == 0:
        raise ValueError("empty series")
    n = len(labels)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for i, label in enumerate(labels):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]
        s = series[label].copy()
        p = s["portfolio"].astype(float).fillna(0)
        p = (1.0 + p).cumprod() if start_from_one else p.cumsum()
        x = pd.to_datetime(s.index)
        ax.plot(x, p.values, color="green", label="Portfolio", linewidth=1.2)
        ax.set_title(f"{label} Gross Return")
        ax.grid(True, alpha=0.3)
        ymin, ymax = float(p.min()), float(p.max())
        if ymin == ymax:
            eps = 1e-4
            ymin -= eps; ymax += eps
        ax.set_ylim(ymin * 0.98, ymax * 1.02)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.margins(x=0.01)
        if i == 0:
            ax.legend(loc="best", fontsize=8)
    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].axis("off")
    fig.tight_layout()
    fig.autofmt_xdate()
    plt.subplots_adjust(hspace=0.35)
    if out_path:
        fig.savefig(out_path, dpi=140)
    return fig
