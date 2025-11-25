#!/usr/bin/env python3
"""
使用训练好的模型对 test集 进行推理。
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

# 使用新的 dataloader 和 model
from .dataloader import get_test_dataloader
from .gru_attention import AttentionGRURes

PathLike = Union[str, Path]

# ================== yml 配置读取 ==================
def load_config(yml_path: str | Path) -> dict:
    yml_path = Path(yml_path)
    with yml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


# ================== 主推理 ==================
def inference_on_test_set(
    model_path: str,
    data_root: str,
    batch_size: int = 256,
    num_workers: int = 4,
    universe: Optional[str] = None,
    seq_len: int = 60,
    hidden_dim: int = 128,
    num_layers: int = 1,
    dropout: float = 0.1,
    device: Union[str, torch.device] = "cuda:0",
    output_file: Optional[str] = None,
):
    # 设备
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        device = torch.device(device)
    else:
        device = torch.device("cpu")
    print(f"推理设备: {device}")

    # dataloader
    test_loader = get_test_dataloader(
        data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        universe=universe,
        seq_len=seq_len,
    )

    if test_loader is None:
        print("无法加载 test dataloader，退出程序。")
        return None

    n_features = test_loader.dataset.n_feat # pyright: ignore[reportAttributeAccessIssue]
    print(f"特征数: {n_features}, 序列长度: {seq_len}")

    test_di = test_loader.dataset.di # pyright: ignore[reportAttributeAccessIssue]
    ii_max = test_loader.dataset.ii # pyright: ignore[reportAttributeAccessIssue]
    print(f"Test set 尺寸: di={test_di}, ii={ii_max}")

    # 全局预测矩阵：全 NaN
    global_preds = np.full((test_di, ii_max), np.nan, dtype=np.float32)

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

    long_pred_list: List[np.ndarray] = []
    long_tgt_list:  List[np.ndarray] = []
    long_di_list:   List[np.ndarray] = []
    long_ii_list:   List[np.ndarray] = []

    # 推理
    model.eval()
    with torch.inference_mode():
        for X_batch, y_batch, di_batch, ii_batch in tqdm(test_loader, desc=f"推理中[test]"):
            X_batch = X_batch.to(device, non_blocking=True)

            y_pred = model(X_batch).view(-1).detach().cpu().numpy().astype(np.float32)
            y_true = y_batch.numpy().reshape(-1).astype(np.float32)
            di_np = di_batch.numpy().astype(np.int64)
            ii_np = ii_batch.numpy().astype(np.int64)

            global_preds[di_np, ii_np] = y_pred

            long_pred_list.append(y_pred)
            long_tgt_list.append(y_true)
            long_di_list.append(di_np)
            long_ii_list.append(ii_np)


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

    # 保存 2D 矩阵
    if output_file is not None:
        out_path = Path(output_file)
        np.save(out_path, global_preds)
        print(f"已保存2D预测矩阵到 {out_path}, shape={global_preds.shape}")

    return out_dict


# ================== 参数解析（只传 config.yml） ==================
def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 AttentionGRURes 对 test set 做日频推理"
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

    inference_on_test_set(
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