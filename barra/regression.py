import numpy as np
import pandas as pd

def _normalize_weights(w: pd.Series) -> pd.Series:
    w = w.astype(float)
    s = w.sum()
    if s == 0:
        return w
    return w / s

def _weighted_mean(df: pd.DataFrame, w: pd.Series) -> pd.Series:
    a = df.to_numpy()
    b = w.to_numpy()[:, None]
    m = (a * b).sum(axis=0)
    return pd.Series(m, index=df.columns)

def _demean_style(df: pd.DataFrame, w: pd.Series) -> pd.DataFrame:
    m = _weighted_mean(df, w)
    return df - m

def _industry_dummies(industry: pd.Series) -> pd.DataFrame:
    g = industry.astype(str)
    return pd.get_dummies(g)

def barra_regression(
    y: pd.Series,
    style: pd.DataFrame,
    industry: pd.Series,
    weights: pd.Series,
    alpha: float = 1e-8,
) -> tuple[pd.Series, pd.Series]:
    idx = y.index
    y = y.loc[idx].astype(float)
    w = _normalize_weights(weights.loc[idx])
    sx = style.loc[idx]
    sx = _demean_style(sx, w)
    d = _industry_dummies(industry.loc[idx]).astype(float)
    cols = list(sx.columns)
    ind_cols = list(d.columns)
    if len(ind_cols) > 0:
        base = ind_cols[-1]
        d_fit = d.drop(columns=[base])
    else:
        d_fit = d
    X = pd.concat([sx, d_fit], axis=1).astype(float)
    Xv = X.to_numpy(dtype=float)
    yv = y.to_numpy()
    ww = w.to_numpy()
    sw = np.sqrt(ww)
    Xw = Xv * sw[:, None]
    yw = yv * sw
    XtX = Xw.T @ Xw
    XtX += alpha * np.eye(XtX.shape[0])
    Xty = Xw.T @ yw
    beta = np.linalg.solve(XtX, Xty)
    b_style = beta[: len(cols)]
    b_ind_fit = beta[len(cols) :]
    fr_style = pd.Series(b_style, index=cols)
    if len(ind_cols) > 0:
        fr_ind = pd.Series(0.0, index=ind_cols)
        for c, v in zip(d_fit.columns, b_ind_fit):
            fr_ind[c] = v
        s_share = (d.T @ w).astype(float)
        sb = s_share[base]
        if sb == 0:
            fr_ind[base] = -float(fr_ind.drop(index=[base]).dot(s_share.drop(index=[base])))
        else:
            fr_ind[base] = -float(fr_ind.drop(index=[base]).dot(s_share.drop(index=[base])) / sb)
    else:
        fr_ind = pd.Series(dtype=float)
    fr = pd.concat([fr_style, fr_ind])
    X_full = pd.concat([sx, d], axis=1).astype(float)
    y_hat = X_full.to_numpy(dtype=float) @ fr.loc[X_full.columns].to_numpy(dtype=float)
    res = pd.Series(yv - y_hat, index=idx)
    return fr, res

def style_regression(
    y: pd.Series,
    style: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 1e-8,
    center: bool = False,
) -> tuple[pd.Series, pd.Series]:
    idx = y.index
    y = y.loc[idx].astype(float)
    w = _normalize_weights(weights.loc[idx])
    X = style.loc[idx].astype(float)
    if center:
        X = _demean_style(X, w)
    Xv = X.to_numpy(dtype=float)
    yv = y.to_numpy()
    sw = np.sqrt(w.to_numpy())
    Xw = Xv * sw[:, None]
    yw = yv * sw
    XtX = Xw.T @ Xw
    XtX += alpha * np.eye(XtX.shape[0])
    Xty = Xw.T @ yw
    beta = np.linalg.solve(XtX, Xty)
    fr_style = pd.Series(beta, index=list(X.columns))
    y_hat = Xv @ fr_style.to_numpy(dtype=float)
    res = pd.Series(yv - y_hat, index=idx)
    return fr_style, res


def panel_factor_regression(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    factor_cols: list[str],
    industry_col: str,
    ret_col: str = "ret",
    weight_col: str = "mcap",
    alpha: float = 1e-8,
):
    fr_rows = []
    resid_parts = []
    for dt, g in df.groupby(date_col):
        sub = g[[asset_col, ret_col, weight_col, industry_col] + factor_cols].dropna()
        sub = sub.set_index(asset_col)
        y = sub[ret_col]
        style = sub[factor_cols]
        industry = sub[industry_col]
        weights = sub[weight_col]
        fr, resid = barra_regression(y, style, industry, weights, alpha)
        fr.name = dt
        fr_rows.append(fr)
        resid.index = pd.MultiIndex.from_product([[dt], resid.index], names=[date_col, asset_col])
        resid_parts.append(resid)
    fr_df = pd.DataFrame(fr_rows).sort_index().fillna(0.0)
    resid_df = pd.concat(resid_parts).to_frame("resid").sort_index()
    return fr_df, resid_df

