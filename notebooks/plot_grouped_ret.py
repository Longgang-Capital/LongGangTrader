"""
日频版（全历史一次性分位分组）：
- alpha = alpha_combo[:,:,k] -> (di,ii)
- 用全历史(沿 axis=0)把每只股票的 alpha 映射到历史百分位(0~100)
- 按百分位分成 20 组
- 对 1d/5d/20d 标签分别计算每组的分组收益，并把3张图画在一个2*2里一次性保存

依赖：
- py3lib.sarray
- numba
- matplotlib
"""

import os
import pathlib
import csv
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange, set_num_threads
from py3lib import sarray

set_num_threads(30)

G = 20
need_plot = True
base_dir = "/local/yjhuang/tscombo_daily_percentile/full_history"

alpha_combo = sarray.load('/nas02/home/yjhuang/qlib/对接本地数据源/alpha158_14tonow_3d.bin').data

label_paths = {
    "1d": "/local/yjhuang/mysapcache/labels/v2v_ret_1d_1128.bin",
    "5d": "/local/yjhuang/mysapcache/labels/v2v_ret_5d_1128.bin",
    "20d": "/local/yjhuang/mysapcache/labels/v2v_ret_20d_1128.bin",
}

valid_candidates = [
    "/local/yjhuang/mysapcache/univ_daily_valid/valid.bin",
    "/local/sap/sapcache_1m_2020_rs/univ_daily_valid/valid.bin",
]

slope_csv_path = "/nas02/home/yjhuang/qlib/对接本地数据源/PPT/slope_fullhistory_1128.csv"


def first_existing_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def load_sarray_first(paths, desc):
    p = first_existing_path(paths)
    if p is None:
        raise FileNotFoundError(
            f"[找不到{desc}] 下面这些路径都不存在：\n" + "\n".join(paths)
        )
    print(f"[加载{desc}] {p}")
    return sarray.load(p).data


@njit(parallel=True)
def percentile_by_rank_numba(train_2d, val_2d):
    Dtr, II = train_2d.shape
    Dva, _  = val_2d.shape
    out = np.full((Dva, II), np.nan, dtype=np.float32)

    for col in prange(II):
        cnt = 0
        for i in range(Dtr):
            if np.isfinite(train_2d[i, col]):
                cnt += 1
        if cnt == 0:
            continue

        buf = np.empty(cnt, dtype=np.float32)
        k = 0
        for i in range(Dtr):
            v = train_2d[i, col]
            if np.isfinite(v):
                buf[k] = v
                k += 1
        buf.sort()

        for i in range(Dva):
            v = val_2d[i, col]
            if not np.isfinite(v):
                continue
            pos = np.searchsorted(buf, v, side="left")  # 0..cnt
            out[i, col] = 100.0 * (pos / cnt)

    return out


@njit(parallel=True)
def grouped_mean_daily(ret_2d, group_id_2d, valid_2d, G):
    Di, II = ret_2d.shape
    out = np.zeros((G, Di), dtype=np.float32)
    cnt = np.zeros((G, Di), dtype=np.int32)

    for di in prange(Di):
        for ii in range(II):
            if valid_2d[di, ii]:
                g = group_id_2d[di, ii]
                if g >= 0:
                    v = ret_2d[di, ii]
                    if not np.isnan(v):
                        out[g, di] += v
                        cnt[g, di] += 1

    for g in prange(G):
        for di in range(Di):
            if cnt[g, di] > 0:
                out[g, di] /= cnt[g, di]
            else:
                out[g, di] = np.nan
    return out, cnt


def slope_from_group_totals(group_totals: np.ndarray) -> float:
    x = np.arange(len(group_totals), dtype=np.float64)
    y = group_totals.astype(np.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    return float(((x - x_mean) * (y - y_mean)).sum() / denom) if denom > 0 else np.nan


def plot_one_horizon(ax, gm, cnt, horizon, slope, G):
    """在一个 ax 里画：20组累计 + LS"""
    colors = plt.cm.viridis(np.linspace(0, 1, G))
    total_cnt = int(cnt.sum())

    # 20 组累计曲线
    for g in range(G):
        curve = np.nancumsum(np.nan_to_num(gm[g], nan=0.0))
        ax.plot(curve, color=colors[g], lw=1.0, alpha=0.95)

    # long-short
    ls = np.nancumsum(np.nan_to_num(gm[G-1], nan=0.0) - np.nan_to_num(gm[0], nan=0.0))
    ax.plot(ls, lw=2.4, color="black", alpha=0.9)

    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.grid(True, alpha=0.25)
    ax.set_title(f"{horizon} | slope={slope:.4g} | cnt={total_cnt}")


def main(alpha_combo: np.ndarray):
    os.makedirs(base_dir, exist_ok=True)

    # ========= 只加载一次 valid / labels =========
    valid_all = load_sarray_first(valid_candidates, desc="valid_ii(daily)").astype(np.bool_)
    rets_all = {k: sarray.load(p).data.astype(np.float32) for k, p in label_paths.items()}

    # ========= CSV 只写一次表头 =========
    csv_file = pathlib.Path(slope_csv_path)
    csv_exists = csv_file.exists()
    if not csv_exists:
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["signal_name", "G", "slope_1d", "slope_5d", "slope_20d"])

    # ========= 遍历每个 alpha_idx =========
    for alpha_idx in range(alpha_combo.shape[-1]):
        alpha = alpha_combo[:, :, alpha_idx].astype(np.float32, copy=False)
        signal_name = f"alpha_combo_f{alpha_idx}_full"
        Di, II = alpha.shape
        print(f"\n[{signal_name}] alpha shape={alpha.shape}")

        if valid_all.shape != alpha.shape:
            raise ValueError(f"valid_all.shape={valid_all.shape} 与 alpha.shape={alpha.shape} 不一致，请检查对齐")

        for k in ["1d", "5d", "20d"]:
            if rets_all[k].shape != alpha.shape:
                raise ValueError(f"{k} label shape={rets_all[k].shape} 与 alpha.shape={alpha.shape} 不一致，请检查对齐")

        # 1) 全历史百分位（无效位置先置 nan）
        alpha_for_pct = alpha.copy()
        alpha_for_pct[~valid_all] = np.nan

        pct = percentile_by_rank_numba(alpha_for_pct, alpha_for_pct)  # (Di, II), 0~100

        # 2) 百分位 -> 组号
        group_id = np.full(pct.shape, -1, dtype=np.int16)
        finite_mask = np.isfinite(pct)
        tmp = (pct[finite_mask] * (G / 100.0)).astype(np.int16)
        tmp[tmp >= G] = G - 1
        group_id[finite_mask] = tmp
        group_id[~valid_all] = -1

        # 3) 计算三种 horizon 的 gm/cnt/slope（先缓存）
        horizons = ["1d", "5d", "20d"]
        gm_dict, cnt_dict, slope_dict = {}, {}, {}

        for horizon in horizons:
            ret_2d = rets_all[horizon].copy()
            ret_2d[~valid_all] = np.nan

            gm, cnt = grouped_mean_daily(ret_2d, group_id, valid_all, G)  # (G, Di)
            gm_dict[horizon] = gm
            cnt_dict[horizon] = cnt

            group_totals = np.array([np.nansum(gm[g]) for g in range(G)], dtype=np.float64)
            slope_dict[horizon] = slope_from_group_totals(group_totals)

            print(f"  [{horizon}] slope={slope_dict[horizon]:.6g} total_cnt={int(cnt.sum())}")

        # 4) 写一行 csv：三种 horizon 的 slope 一起写
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([signal_name, G, slope_dict["1d"], slope_dict["5d"], slope_dict["20d"]])

        # 5) 2x2 一次性画并保存
        if need_plot:
            png = os.path.join(base_dir, f"{signal_name}_3in1_groups{G}.png")
            if os.path.exists(png):
                print(f"[跳过] 已存在：{png}")
                continue

            fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)

            plot_one_horizon(axes[0, 0], gm_dict["1d"],  cnt_dict["1d"],  "1d",  slope_dict["1d"],  G)
            plot_one_horizon(axes[0, 1], gm_dict["5d"],  cnt_dict["5d"],  "5d",  slope_dict["5d"],  G)
            plot_one_horizon(axes[1, 0], gm_dict["20d"], cnt_dict["20d"], "20d", slope_dict["20d"], G)

            # 右下角：放 legend + 文字说明（不画曲线）
            ax_leg = axes[1, 1]
            ax_leg.axis("off")

            colors = plt.cm.viridis(np.linspace(0, 1, G))
            handles = []
            labels = []
            total_cnt = int(cnt_dict["1d"].sum())
            for g in range(G):
                # 用 1d 的占比当展示（你也可以改成 20d 或三者平均）
                w_cnt = int(cnt_dict["1d"][g].sum())
                ratio = (w_cnt / total_cnt) if total_cnt > 0 else 0.0
                h, = ax_leg.plot([], [], color=colors[g], lw=2)
                handles.append(h)
                labels.append(f"G{g} ({ratio:.1%})")

            h_ls, = ax_leg.plot([], [], color="black", lw=3)
            handles.append(h_ls)
            labels.append("LS (G19 - G0)")

            ax_leg.legend(handles, labels, loc="center", ncol=2, frameon=False, fontsize=9)
            ax_leg.text(
                0.5, 0.02,
                f"{signal_name}\nG={G}\nslopes: 1d={slope_dict['1d']:.4g}, 5d={slope_dict['5d']:.4g}, 20d={slope_dict['20d']:.4g}",
                ha="center", va="bottom", fontsize=11
            )

            fig.suptitle(f"{signal_name} | full-history TS-percentile grouping", fontsize=16, y=1.02)
            plt.savefig(png, dpi=200)
            plt.close()
            print(f"[保存] 2x2 合图 -> {png}")

    print("\n全部完成 ✅")


if __name__ == "__main__":
    main(alpha_combo)
