import numpy as np
import pandas as pd
from regression import barra_regression, style_regression, _normalize_weights, _demean_style
from utils import wide_weights_to_long
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


class Barra:
    def __init__(self, factor_cols: list[str], industry_col: str | None = None, ret_col: str = "ret", mcap_col: str = "mcap", style_only: bool = False):
        self.factor_cols = factor_cols
        self.industry_col = industry_col
        self.ret_col = ret_col
        self.mcap_col = mcap_col
        self.style_only = style_only

    def run_one_day(
        self,
        df: pd.DataFrame,
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series,
        alpha: float = 1e-8,
    ) -> pd.DataFrame:
        df = df.copy()
        idx = df.index
        y = df[self.ret_col]
        w_reg = df[self.mcap_col]
        style = df[self.factor_cols]
        industry = df[self.industry_col]
        if self.style_only or self.industry_col is None:
            fr, resid = style_regression(y, style, w_reg, alpha, center=False)
        else:
            fr, resid = barra_regression(y, style, industry, w_reg, alpha)
        pw = _normalize_weights(portfolio_weights.loc[idx].fillna(0))
        bw = _normalize_weights(benchmark_weights.loc[idx].fillna(0))
        style_center = _demean_style(style, _normalize_weights(w_reg))
        exp_port_style = (style_center.T @ pw).loc[self.factor_cols]
        exp_bm_style = (style_center.T @ bw).loc[self.factor_cols]
        rows = []
        for f in self.factor_cols:
            e_port = float(exp_port_style.get(f, 0.0))
            e_bm = float(exp_bm_style.get(f, 0.0))
            rel = e_port - e_bm
            fr_f = float(fr.get(f, 0.0))
            contrib = rel * fr_f
            rows.append([
                f,
                f,
                e_port,
                e_bm,
                rel,
                fr_f,
                contrib,
                float((pw * resid).sum()),
                float((bw * resid).sum()),
                float((pw * resid).sum() - (bw * resid).sum()),
                contrib + float((pw * resid).sum() - (bw * resid).sum()),
            ])
        df_out = pd.DataFrame(rows, columns=[
            "行业代码",
            "行业名称",
            "资产组合暴露",
            "基准暴露",
            "相对暴露",
            "因子收益",
            "因子收益贡献",
            "资产组合残差收益",
            "基准残差收益",
            "残差贡献",
            "总贡献",
        ])
        total_factor = float(df_out["因子收益贡献"].replace(np.nan, 0.0).sum())
        total_resid = float(df_out["残差贡献"].replace(np.nan, 0.0).sum() - total_factor)
        active_return = float((pw * y).sum() - (bw * y).sum())
        rows_total = [[
            "汇总",
            "汇总",
            np.nan,
            np.nan,
            np.nan,
            total_factor,
            total_factor,
            np.nan,
            np.nan,
            active_return - total_factor,
            active_return,
        ]]
        df_sum = pd.DataFrame(rows_total, columns=df_out.columns)
        df_out = pd.concat([df_out, df_sum], ignore_index=True)
        return df_out

    def run(
        self,
        df: pd.DataFrame,
        portfolio_weights: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        date_col: str,
        alpha: float = 1e-8,
    ) -> dict:
        res = {}
        it = df.groupby(date_col)
        total = int(df[date_col].nunique()) if date_col in df.columns else None
        for dt, g in tqdm(it, total=total):
            pw = portfolio_weights[portfolio_weights[date_col] == dt].set_index("asset")["weight"]
            bw = benchmark_weights[benchmark_weights[date_col] == dt].set_index("asset")["weight"]
            sub = g.set_index("asset")
            res[dt] = self.run_one_day(sub, pw, bw, alpha)
        return res
    def contribution(
        self,
        df: pd.DataFrame,
        portfolio_wide: pd.DataFrame,
        benchmark_wide: pd.DataFrame | None,
        date_col: str,
        asset_col: str,
        alpha: float = 1e-8,
    ) -> dict[str, pd.DataFrame]:
        pw_long = wide_weights_to_long(portfolio_wide, date_col)
        bw_long = wide_weights_to_long(benchmark_wide, date_col) if benchmark_wide is not None else None
        sums = pw_long.groupby(date_col)["weight"].apply(lambda x: float(np.abs(x).sum()))
        valid_dates = set(sums[sums > 1e-12].index)
        out: dict[str, list] = {}
        for dt, g in df.groupby(date_col):
            if dt not in valid_dates:
                continue
            pw = pw_long[pw_long[date_col] == dt].set_index("asset")["weight"]
            bw = bw_long[bw_long[date_col] == dt].set_index("asset")["weight"] if bw_long is not None else None
            sub = g.set_index(asset_col)
            if bw is None:
                fr, resid = style_regression(sub[self.ret_col], sub[self.factor_cols], sub[self.mcap_col], alpha, center=False)
                for f in self.factor_cols:
                    label = str(f)
                    if label not in out:
                        out[label] = []
                    e_port = float((sub[f] * pw).sum())
                    contrib = e_port * float(fr.get(f, 0.0))
                    out[label].append((dt, contrib))
            else:
                res_day = self.run_one_day(sub, pw, bw, alpha)
                for _, row in res_day.iterrows():
                    label = str(row["行业名称"])  # 同行业与风格的列名
                    port = float(row["资产组合暴露"]) * float(row["因子收益"]) if not pd.isna(row["资产组合暴露"]) else float(row["因子收益贡献"])  # 风格行
                    bench = float(row["基准暴露"]) * float(row["因子收益"]) if not pd.isna(row["基准暴露"]) else float(row["因子收益贡献"])  # 风格行
                    if not pd.isna(row["资产组合残差收益"]) and not pd.isna(row["基准残差收益"]):
                        port += float(row["资产组合残差收益"])  # 行业加残差
                        bench += float(row["基准残差收益"])   # 行业加残差
                    active = port - bench
                    if label not in out:
                        out[label] = []
                    out[label].append((dt, port, bench, active))
        series = {}
        for label, rows in out.items():
            if benchmark_wide is None:
                s = pd.DataFrame(rows, columns=[date_col, "portfolio"]).sort_values(date_col)
            else:
                s = pd.DataFrame(rows, columns=[date_col, "portfolio", "benchmark", "active"]).sort_values(date_col)
            s = s.set_index(date_col)
            series[label] = s
        return series



