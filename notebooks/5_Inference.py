#!/usr/bin/env python3
"""
使用训练好的模型做推理，基于 DailyAlphaDataloader 里的日频 dataloader：
- 对 train / val / test 都跑一遍
- 生成一张全局的 (sum(di), ii_max) 的预测矩阵，未出现的位置是 NaN
- 同时导出 axis=0 的长表 [pred, target, di_global, ii]
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm

# 你自己的 dataloader
from DailyAlphaDataloader import get_infer_dataloaders  # 日频的那个版本
from py3lib import taskbase, sarray

# ================== 全局配置 ==================
task = taskbase.TaskBase()
task.get_options({'sap_cache_dir': '/local/sap/sapcache_1m_2020_rs'})


# ================== 模型定义（跟训练保持一致） ==================
class AttentionGRURes(nn.Module):
    """
    结构与研报一致的轻量Attention + GRU + Res + FFN：
    """
    def __init__(self, input_dim=158, hidden_dim=128, num_layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn_score = nn.Linear(hidden_dim, 1)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: (B, T, F)
        h, _ = self.gru(x)                      # (B, T, H)
        attn_logits = self.attn_score(h)        # (B, T, 1)
        attn_w = torch.softmax(attn_logits, 1)  # (B, T, 1)
        ctx = (attn_w * h).sum(1)               # (B, H)
        ctx = self.ln1(ctx + self.dropout(self.ffn(ctx)))
        out = self.head(ctx).squeeze(-1)        # (B,)
        return out


# ================== 单个 loader 推理并填进“全局大矩阵” ==================
def _infer_on_loader_into_global(
    dataloader,
    model: nn.Module,
    device: torch.device,
    split_name: str,
    global_preds: np.ndarray,
    di_offset: int,
    long_pred_list: List[np.ndarray],
    long_tgt_list: List[np.ndarray],
    long_di_list: List[np.ndarray],
    long_ii_list: List[np.ndarray],
):
    """
    dataloader: 返回 (X, y, di, ii)
    global_preds: (total_di, ii_max) 的大矩阵
    di_offset: 当前 split 的 di 在全局里的起始位置
    其余四个 list 用来拼“长表”
    """
    model.eval()
    _di = dataloader.dataset.di
    _ii = dataloader.dataset.ii
    print(f"[{split_name}] 本split尺寸: di={_di}, ii={_ii}, di_offset={di_offset}")

    with torch.inference_mode():
        for X_batch, y_batch, di_batch, ii_batch in tqdm(dataloader, desc=f"推理中[{split_name}]"):
            X_batch = X_batch.to(device, non_blocking=True)

            # (B,)
            y_pred = model(X_batch).view(-1).detach().cpu().numpy().astype(np.float32)


            # 转成 numpy
            y_true = y_batch.numpy().reshape(-1).astype(np.float32)
            di_np = di_batch.numpy().astype(np.int64) + di_offset   # 全局 di
            ii_np = ii_batch.numpy().astype(np.int64)

            print(f"{y_pred=}, {di_np=}, {ii_np=}")

            # 填进全局大矩阵里
            global_preds[di_np, ii_np] = y_pred

            # 收集长表
            long_pred_list.append(y_pred)
            long_tgt_list.append(y_true)
            long_di_list.append(di_np)
            long_ii_list.append(ii_np)


# ================== 主推理 ==================
def inference_all_splits(
    model_path: str,
    data_root: str,
    batch_size: int = 256,
    num_workers: int = 4,
    universe: Optional[str] = None,
    seq_len: int = 60,
    hidden_dim: int = 128,
    num_layers: int = 1,
    dropout: float = 0.1,
    device: str = "cuda:0",
    output_file: Optional[str] = None,
):
    # 设备
    if torch.cuda.is_available() and device.startswith("cuda"):
        device = torch.device(device)
    else:
        device = torch.device("cpu")
    print(f"推理设备: {device}")

    # 拿到 train/val/test loader（都是你日频那个） :contentReference[oaicite:3]{index=3}
    train_loader, val_loader, test_loader = get_infer_dataloaders(
        data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        universe=universe,
        seq_len=seq_len,
    )

    # 拿特征数
    if train_loader is not None:
        n_features = train_loader.dataset.n_feat
    elif val_loader is not None:
        n_features = val_loader.dataset.n_feat
    else:
        n_features = test_loader.dataset.n_feat
    print(f"特征数: {n_features}, 序列长度: {seq_len}")

    # 各 split 的 di、ii
    train_di = train_loader.dataset.di if train_loader is not None else 0
    val_di   = val_loader.dataset.di   if val_loader is not None else 0
    test_di  = test_loader.dataset.di  if test_loader is not None else 0

    # ii 最好取三个里的最大值，保险
    ii_max = 0
    for ld in [train_loader, val_loader, test_loader]:
        if ld is not None:
            ii_max = max(ii_max, ld.dataset.ii)
    total_di = train_di + val_di + test_di

    print(f"全局尺寸: di={total_di}, ii={ii_max}")

    # 全局预测矩阵：全 NaN
    global_preds = np.full((total_di, ii_max), np.nan, dtype=np.float32)

    # 创建模型并加载权重
    model = AttentionGRURes(
        input_dim=n_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    print(f"加载模型: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("模型加载完毕")

    # 这几个 list 用来拼 axis=0 的“长表”
    long_pred_list: List[np.ndarray] = []
    long_tgt_list:  List[np.ndarray] = []
    long_di_list:   List[np.ndarray] = []
    long_ii_list:   List[np.ndarray] = []

    # 1) train
    di_offset = 0
    if train_loader is not None:
        _infer_on_loader_into_global(
            train_loader, model, device, "train",
            global_preds, di_offset,
            long_pred_list, long_tgt_list, long_di_list, long_ii_list,
        )
        di_offset += train_di  # 下一个 split 往后排

    # 2) val
    if val_loader is not None:
        _infer_on_loader_into_global(
            val_loader, model, device, "val",
            global_preds, di_offset,
            long_pred_list, long_tgt_list, long_di_list, long_ii_list,
        )
        di_offset += val_di

    # 3) test
    if test_loader is not None:
        _infer_on_loader_into_global(
            test_loader, model, device, "test",
            global_preds, di_offset,
            long_pred_list, long_tgt_list, long_di_list, long_ii_list,
        )

    # ========== 拼长表 ==========
    preds_flat = np.concatenate(long_pred_list, axis=0)
    tgts_flat  = np.concatenate(long_tgt_list, axis=0)
    di_flat    = np.concatenate(long_di_list, axis=0)
    ii_flat    = np.concatenate(long_ii_list, axis=0)

    # 打个包方便调试
    out_dict = {
        "preds_flat": preds_flat,
        "tgts_flat": tgts_flat,
        "di_flat": di_flat,
        "ii_flat": ii_flat,
        "preds_2d": global_preds,
    }

    # 保存
    if output_file is not None:
        out_path = Path(output_file)
        # 1) 保存长表
        # stacked = np.stack(
        #     [preds_flat, tgts_flat, di_flat.astype(np.float32), ii_flat.astype(np.float32)],
        #     axis=1,
        # )  # (N, 4)
        # sarray.save_ndarray(stacked, out_path)
        # print(f"已保存长表到 {out_path}, shape={stacked.shape}")

        # 2) 再存一个 2D 的
        # mat_path = out_path.with_name(out_path.stem + "_by_di_ii" + out_path.suffix)
        sarray.save_ndarray(global_preds, output_file)
        print(f"已保存2D预测矩阵到 {output_file}, shape={global_preds.shape}")

    return out_dict


def main():
    # 按你 task.options 的风格来一版
    data_root = task.options.path_root
    if torch.cuda.is_available():
        device = f"cuda:{int(task.options.gpu)}"
    else:
        device = "cpu"

    inference_all_splits(
        model_path=task.options.model_path,
        data_root=data_root,
        batch_size=task.options.batch_size,
        num_workers=task.options.num_workers,
        universe=getattr(task.options, "universe", None),
        seq_len=getattr(task.options, "seq_len", 60),
        hidden_dim=getattr(task.options, "hidden_dim", 128),
        num_layers=getattr(task.options, "num_layers", 1),
        dropout=getattr(task.options, "dropout", 0.1),
        device=device,
        output_file=getattr(task.options, "output_file", None),
    )


if __name__ == "__main__":
    main()
