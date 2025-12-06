import os
import pandas as pd
from barra import Barra
from regression import panel_factor_regression

def _norm_code(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    s = s.str.replace('.XSHE', '.SZ', regex=False)
    s = s.str.replace('.XSHG', '.SH', regex=False)
    return s

def _detect_value_col(df: pd.DataFrame, exclude: list[str]) -> str:
    for c in df.columns:
        if c not in exclude:
            return c
    return df.columns[-1]

def _detect_code_col(df: pd.DataFrame, hint: str | None = None) -> str:
    if hint and hint in df.columns:
        return hint
    candidates = [
        'order_book_id', 'code', 'asset', 'ticker', 'sec_code', 'stock_id', 'symbol'
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[1]

def _to_date(s: pd.Series) -> pd.Series:
    d1 = pd.to_datetime(s, errors='coerce')
    d2 = pd.to_datetime(s.astype(str).str.zfill(8), format='%Y%m%d', errors='coerce')
    d = d1.fillna(d2)
    return d.dt.date

def _ensure_date_column(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if date_col in df.columns:
        d = _to_date(df[date_col])
        out = df.copy()
        out[date_col] = d
        return out
    elif df.index.name is not None:
        d = _to_date(df.index.to_series())
        out = df.reset_index()
        out = out.rename(columns={out.columns[0]: date_col})
        out[date_col] = d
        return out
    else:
        d = _to_date(df.iloc[:, 0])
        out = df.copy()
        out.insert(0, date_col, d)
        return out

def _clean_asset(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip()
    x = x.replace({"": pd.NA, "None": pd.NA, "nan": pd.NA, "NaN": pd.NA})
    return x

def _is_one_hot(df: pd.DataFrame) -> bool:
    if df.shape[1] <= 1:
        return False
    try:
        vals = df.apply(pd.to_numeric, errors='coerce')
    except Exception:
        return False
    uniq = pd.unique(vals.values.ravel())
    uniq = [u for u in uniq if pd.notna(u)]
    return set(uniq).issubset({0, 1})

def run_barra_from_dfs(
    df_factors: pd.DataFrame,
    df_industry: pd.DataFrame,
    df_returns: pd.DataFrame,
    df_pw_wide: pd.DataFrame,
    df_bw_wide: pd.DataFrame | int | float | None,
    df_mcap: pd.DataFrame | None = None,
    date_col: str = 'date',
    code_col: str = 'order_book_id',
    out_path: str | None = 'outputs/barra_grid.png',
    benchmark_to_ones: bool | None = None,
    debug: bool = False,
    return_detail: bool = False,
    no_benchmark: bool = False,
):
    f = _ensure_date_column(df_factors.copy(), date_col)
    code_f = _detect_code_col(f, code_col)
    f['asset'] = _norm_code(_clean_asset(f[code_f]))
    f = f.dropna(subset=['asset'])
    factor_cols = [c for c in f.columns if c not in [date_col, code_f, 'asset']]
    num_cols = []
    for c in factor_cols:
        if pd.api.types.is_numeric_dtype(f[c]):
            num_cols.append(c)
        else:
            try:
                f[c] = pd.to_numeric(f[c], errors='coerce')
                num_cols.append(c)
            except Exception:
                pass
    factor_cols = num_cols
    f = f[[date_col, 'asset'] + factor_cols]
    ind = _ensure_date_column(df_industry.copy(), date_col)
    code_i = _detect_code_col(ind, code_col)
    ind['asset'] = _norm_code(_clean_asset(ind[code_i]))
    ind = ind.dropna(subset=['asset'])
    other_i = [c for c in ind.columns if c not in [date_col, code_i, 'asset']]
    if _is_one_hot(ind[other_i]):
        dummies = ind[other_i].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        label = dummies.idxmax(axis=1)
        label = label.where(dummies.sum(axis=1) > 0, 'Unknown')
        ind = ind[[date_col, 'asset']].assign(industry=label.astype(str))
    else:
        ind_col = _detect_value_col(ind, [date_col, code_i, 'asset'])
        ind = ind[[date_col, 'asset', ind_col]].rename(columns={ind_col: 'industry'})
        ind['industry'] = ind['industry'].astype(str)
    r = _ensure_date_column(df_returns.copy(), date_col)
    code_r = _detect_code_col(r, code_col)
    r['asset'] = _norm_code(_clean_asset(r[code_r]))
    r = r.dropna(subset=['asset'])
    ret_col = 'ret' if 'ret' in r.columns else _detect_value_col(r, [date_col, code_r, 'asset'])
    r = r[[date_col, 'asset', ret_col]].rename(columns={ret_col: 'ret'})
    r['ret'] = pd.to_numeric(r['ret'], errors='coerce')
    r = r.dropna(subset=['ret'])
    if df_mcap is not None:
        m = _ensure_date_column(df_mcap.copy(), date_col)
        code_m = _detect_code_col(m, code_col)
        m['asset'] = _norm_code(_clean_asset(m[code_m]))
        m = m.dropna(subset=['asset'])
        mc_candidates = ['mcap', 'market_cap', 'float_mcap', 'circulating_mcap', 'market_cap_2']
        mc_col = next((c for c in mc_candidates if c in m.columns), None)
        if mc_col is None:
            mc_col = _detect_value_col(m, [date_col, code_m, 'asset'])
        m = m[[date_col, 'asset', mc_col]].rename(columns={mc_col: 'mcap'})
        m['mcap'] = pd.to_numeric(m['mcap'], errors='coerce')
        m = m.dropna(subset=['mcap'])
    else:
        m = r[[date_col, 'asset']].copy()
        m['mcap'] = 1.0
    if debug:
        print('factors shape', f.shape)
        print('industry shape', ind.shape)
        print('mcap shape', m.shape)
        print('returns shape', r.shape)
    panel = f.merge(ind, on=[date_col, 'asset']).merge(m, on=[date_col, 'asset']).merge(r, on=[date_col, 'asset']).sort_values([date_col, 'asset'])
    if panel.empty:
        raise ValueError('panel is empty after merging; check date/code alignment')
    pw = _ensure_date_column(df_pw_wide.copy(), date_col)
    if date_col not in pw.columns:
        pw = pw.rename(columns={pw.columns[0]: date_col})
    other_cols = [c for c in pw.columns if c != date_col]
    norm_cols = _norm_code(pd.Series(other_cols)).tolist()
    rename_map = {old: new for old, new in zip(other_cols, norm_cols)}
    pw = pw.rename(columns=rename_map)
    if no_benchmark:
        bw = None
    elif benchmark_to_ones is None:
        benchmark_to_ones = isinstance(df_bw_wide, (int, float)) and float(df_bw_wide) == 1.0
    if benchmark_to_ones:
        bw = pd.DataFrame({date_col: pw[date_col]})
        for c in [col for col in pw.columns if col != date_col]:
            bw[c] = 1.0
    else:
        bw = (df_bw_wide.copy() if df_bw_wide is not None else pw.copy())
        bw = _ensure_date_column(bw, date_col)
        if date_col not in bw.columns:
            bw = bw.rename(columns={bw.columns[0]: date_col})
        other_cols_b = [c for c in bw.columns if c != date_col]
        norm_cols_b = _norm_code(pd.Series(other_cols_b)).tolist()
        rename_map_b = {old: new for old, new in zip(other_cols_b, norm_cols_b)}
        bw = bw.rename(columns=rename_map_b)
    pw[date_col] = _to_date(pw[date_col])
    if bw is not None:
        bw[date_col] = _to_date(bw[date_col])
    panel_dates = sorted(panel[date_col].unique())
    pw_idx = pw.set_index(date_col).sort_index()
    pw_idx = pw_idx.reindex(panel_dates).ffill().bfill()
    pw = pw_idx.reset_index().rename(columns={"index": date_col})
    if bw is not None:
        bw_idx = bw.set_index(date_col).sort_index()
        bw_idx = bw_idx.reindex(panel_dates).ffill().bfill()
        bw = bw_idx.reset_index().rename(columns={"index": date_col})
    panel = panel[panel[date_col].isin(panel_dates)]
    atb = BarraAttributor(factor_cols, 'industry', ret_col='ret', mcap_col='mcap')
    series = atb.contribution_series(panel, pw, bw, date_col=date_col, asset_col='asset')
    if len(series) == 0:
        series_total = {}
        rows = []
        for dt, sub in panel.groupby(date_col):
            ret = sub.set_index('asset')['ret']
            row_p = pw[pw[date_col] == dt]
            row_b = bw[bw[date_col] == dt]
            if row_p.empty or row_b.empty:
                continue
            wp = row_p.drop(columns=[date_col]).iloc[0]
            wb = row_b.drop(columns=[date_col]).iloc[0]
            wp = wp.reindex(ret.index, fill_value=0).astype(float)
            wb = wb.reindex(ret.index, fill_value=0).astype(float)
            sp = wp.sum(); sb = wb.sum()
            if sp != 0: wp = wp / sp
            if sb != 0: wb = wb / sb
            port = float((wp * ret).sum())
            bench = float((wb * ret).sum())
            rows.append((dt, port, bench, port - bench))
        if rows:
            df_total = pd.DataFrame(rows, columns=[date_col, 'portfolio', 'benchmark', 'active']).sort_values(date_col).set_index(date_col)
        series_total['Active Portfolio Gross Return'] = df_total
        series = series_total
    if out_path is not None:
        from plotting import plot_contribution_grid
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if no_benchmark:
            from plotting import plot_factor_grid
            plot_factor_grid(series, ncols=2, out_path=out_path, start_from_one=True, drop_last=True)
        else:
            plot_contribution_grid(series, ncols=2, out_path=out_path, start_from_one=True)
    if return_detail:
        fr_df, resid_df = panel_factor_regression(
            panel.rename(columns={'asset': 'asset'}),
            date_col=date_col,
            asset_col='asset',
            factor_cols=factor_cols,
            industry_col='industry',
            ret_col='ret',
            weight_col='mcap',
        )
        return series, {'fr_df': fr_df, 'resid_df': resid_df}
    return series
