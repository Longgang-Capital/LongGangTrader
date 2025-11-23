#!/usr/bin/env python3
"""
使用训练好的模型做推理，基于 DailyAlphaDataloader 里的日频 dataloader：
- 对 train / val / test 都跑一遍
- 生成一张全局的 (sum(di), ii_max) 的预测矩阵，未出现的位置是 NaN
- 同时导出 axis=0 的长表 [pred, target, di_global, ii]
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Union

import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import json
import os
import argparse
import yaml

# 你自己的 dataloader
from DailyAlphaDataloader import get_infer_dataloaders  # 日频的那个版本

PathLike = Union[str, Path]

# ================== yml 配置读取 ==================
def load_config(yml_path: str | Path) -> dict:
    yml_path = Path(yml_path)
    with yml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


# ========= 1. dtype 解析（给 load_sarray_data 用，保留） =========
def _conv_dtype(_dtype):

    if isinstance(_dtype, str):
        return _dtype
    elif isinstance(_dtype, list):
        if len(_dtype) == 2:
            if isinstance(_dtype[1], int):
                return (_conv_dtype(_dtype[0]), _dtype[1])
            elif isinstance(_dtype[1], list) and _dtype[1] and isinstance(_dtype[1][0], int):
                return (_conv_dtype(_dtype[0]), tuple(_dtype[1]))
        # 结构化 dtype
        ret = []
        for field in _dtype:
            if len(field) == 3:
                ret.append((
                    field[0],
                    _conv_dtype(field[1]),
                    field[2] if isinstance(field[2], int) else tuple(field[2])
                ))
            elif len(field) == 2:
                ret.append((field[0], _conv_dtype(field[1])))
            else:
                raise Exception(f"failed to parse dtype: {_dtype}")
        return ret
    else:
        raise Exception(f"failed to parse dtype: {_dtype}")


# ========= 2. 带 !include 的 json 加载（给 load_sarray_data 用，保留） =========
class _JsonLoader:
    """支持 !include 语法的 json loader，避免递归 include 死循环"""

    def __init__(self) -> None:
        self._loaded = set()

    def load(self, jsonfile: PathLike) -> dict:
        p = Path(jsonfile)
        data = json.load(p.open('r', encoding='utf8'))
        self._loaded.add(str(p.resolve()))

        if '!include' in data:
            inc = data.pop('!include')
            inc_path = Path(inc) if os.path.isabs(inc) else p.parent / inc
            inc_abs = str(inc_path.resolve())

            if inc_abs not in self._loaded:
                self._loaded.add(inc_abs)
                parent_data = self.load(inc_path)
                parent_data.update(data)
                data = parent_data

        return data


def load_json(filename: PathLike) -> dict:
    loader = _JsonLoader()
    return loader.load(filename)

def load_sarray_data(datafile: PathLike,
                     copy_on_write: bool = True,
                     offset: int = 0) -> np.ndarray:
    datafile = Path(datafile)
    jsonfile = Path(str(datafile) + '.json')

    if not jsonfile.exists():
        raise FileNotFoundError(f"json file not found: {jsonfile}")

    attr = load_json(jsonfile)

    # dtype
    raw_dtype = attr.get('dtype')
    if raw_dtype is None:
        raise ValueError(f"`dtype` not found in {jsonfile}")
    dtype_align = attr.get('dtype_align', True)
    dtype = np.dtype(_conv_dtype(raw_dtype), align=dtype_align)

    # shape
    shape: List[int] = attr.get('shape', [])
    real_offset = offset or attr.get('offset', 0)

    if not shape:
        fsize = datafile.stat().st_size - real_offset
        if fsize % dtype.itemsize != 0:
            raise ValueError(f"file size {fsize} not divisible by itemsize {dtype.itemsize}")
        shape = [fsize // dtype.itemsize]

    mode = 'c' if copy_on_write else 'r'
    arr = np.memmap(datafile, dtype=dtype, mode=mode,
                    shape=tuple(shape), offset=real_offset)
    return arr


# ================== 模型定义（跟训练保持一致） ==================
class AttentionGRURes(nn.Module):
    """
    Attention + GRU + Res + FFN：
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

    model.eval()
    _di = dataloader.dataset.di
    _ii = dataloader.dataset.ii
    print(f"[{split_name}] 本split尺寸: di={_di}, ii={_ii}, di_offset={di_offset}")

    with torch.inference_mode():
        for X_batch, y_batch, di_batch, ii_batch in tqdm(dataloader, desc=f"推理中[{split_name}]"):
            X_batch = X_batch.to(device, non_blocking=True)

            y_pred = model(X_batch).view(-1).detach().cpu().numpy().astype(np.float32)
            y_true = y_batch.numpy().reshape(-1).astype(np.float32)
            di_np = di_batch.numpy().astype(np.int64) + di_offset   # 全局 di
            ii_np = ii_batch.numpy().astype(np.int64)

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

    # dataloader
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

    # ii 取三个里的最大值
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
        di_offset += train_di

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

    # ========== 拼长表（内存里用，暂时不存盘）==========
    preds_flat = np.concatenate(long_pred_list, axis=0)
    tgts_flat  = np.concatenate(long_tgt_list, axis=0)
    di_flat    = np.concatenate(long_di_list, axis=0)
    ii_flat    = np.concatenate(long_ii_list, axis=0)

    out_dict = {
        "preds_flat": preds_flat,
        "tgts_flat": tgts_flat,
        "di_flat": di_flat,
        "ii_flat": ii_flat,
        "preds_2d": global_preds,
    }

    # 保存 2D 矩阵（完全不依赖 sarray，用 np.save）
    if output_file is not None:
        out_path = Path(output_file)
        np.save(out_path, global_preds)
        print(f"已保存2D预测矩阵到 {out_path}, shape={global_preds.shape}")

    return out_dict


# ================== 参数解析（只传 config.yml） ==================
def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 AttentionGRURes 对 train/val/test 做日频推理（yml 配置版）"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="推理配置 yml 路径"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # 从 yml 中取参数，提供合理默认值
    model_path   = cfg["model_path"]
    data_root    = cfg["path_root"]
    batch_size   = cfg.get("batch_size", 256)
    num_workers  = cfg.get("num_workers", 4)
    universe     = cfg.get("universe", None)
    seq_len      = cfg.get("seq_len", 60)
    hidden_dim   = cfg.get("hidden_dim", 128)
    num_layers   = cfg.get("num_layers", 1)
    dropout      = cfg.get("dropout", 0.1)
    output_file  = cfg.get("output_file", None)

    # device / gpu
    if "device" in cfg:
        device = cfg["device"]
    elif "gpu" in cfg and torch.cuda.is_available():
        device = f"cuda:{int(cfg['gpu'])}"
    else:
        device = "cpu"

    inference_all_splits(
        model_path=model_path,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        universe=universe,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        device=device,
        output_file=output_file,
    )


if __name__ == "__main__":
    main()
