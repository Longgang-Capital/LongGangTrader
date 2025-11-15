#!/usr/bin/env python3
import sys
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import bottleneck as bn
from tqdm.auto import tqdm
from torchinfo import summary
import mlflow
import mlflow.pytorch

from DailyAlphaDataloader import get_dataloaders 
from py3lib import taskbase

sys.path.append('/nas02/sap/ver/stable')  # stable是最新的稳定版本
from sap.base import sapa_rust

# ================== 全局配置 ==================
task = taskbase.TaskBase()
task.get_options({'sap_cache_dir': '/local/sap/sapcache_1m_2020_rs'})

MLFLOW_CONFIG = {
    'tracking_uri': "http://127.0.0.1:5000/",
    'experiment_name': "AttentionGRU"
}

# ========= AttentionGRU(Res) 模型定义 =========
class AttentionGRURes(nn.Module):
    """
    结构与研报一致的轻量Attention + GRU + Res + FFN：
    - GRU 提取时序特征
    - 简单自注意力对各时间步加权
    - FFN + 残差 + LayerNorm
    - 输出一维回归值
    """
    def __init__(self, input_dim=158, hidden_dim=128, num_layers=1,
                 dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 简化注意力：每个时间步 h_t → 标量 score_t
        self.attn_score = nn.Linear(hidden_dim, 1)

        # 残差前馈
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # LayerNorm 稳定训练
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, F)
        gru_out, _ = self.gru(x)          # (B, T, H)

        # 1) 注意力打分
        attn_logits = self.attn_score(gru_out)            # (B, T, 1)
        attn_weights = torch.softmax(attn_logits, dim=1)  # (B, T, 1)

        # 2) 加权得到全局时序向量
        ctx = (attn_weights * gru_out).sum(dim=1)         # (B, H)

        # 3) FFN + 残差 + LN
        x1 = self.ln1(ctx + self.dropout(self.ffn(ctx)))  # (B, H)

        # 4) 一维回归输出
        out = self.head(x1)                               # (B, 1)
        return out.squeeze(-1)                            # (B,)


# ========= 早停 =========
class EarlyStopping:
    """早停机制，当验证指标连续不改善时停止训练
    支持 mode='min' 或 'max'，这里我们会用 'max' 来跑 IC
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-6, mode: str = "min"):
        assert mode in ("min", "max")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        if mode == "min":
            self.best_score = float('inf')
        else:
            self.best_score = -float('inf')
        self.best_model_state = None

    def __call__(self, score: float, model) -> bool:
        """返回True表示应该停止训练"""
        if math.isnan(score):
            # NaN 视为极差，不更新 best，但计入 patience
            self.counter += 1
            return self.counter >= self.patience

        improved = False
        if self.mode == "min":
            if score < self.best_score - self.min_delta:
                improved = True
        else:  # mode == "max"
            if score > self.best_score + self.min_delta:
                improved = True

        if improved:
            self.best_score = score
            self.counter = 0
            if model is not None:
                self.best_model_state = model.state_dict()
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best_model(self, model):
        """恢复到最佳模型状态"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


# ========= 评估一个 dataloader，算 IC / winrate 等 =========
def loop_dataloader_get_perf(model, dataloader, device, loss_fn):
    model.eval()
    D, I = dataloader.dataset.di, dataloader.dataset.ii  # 天数 & 股票数

    # 用 (D, I) 存每个 (di, ii) 的真实值和预测值
    y_true_mat = np.full((D, I), np.nan, dtype=np.float32)
    y_pred_mat = np.full((D, I), np.nan, dtype=np.float32)

    total_loss = 0.0
    total_count = 0

    with torch.inference_mode(), tqdm(total=len(dataloader),
                                      unit="batch",
                                      desc=f"{dataloader.dataset.split} eval:") as pbar:
        for X_batch, y_batch, di_batch, ii_batch in dataloader:
            # X: (B, T, F)
            X_batch = X_batch.to(device, non_blocking=True)
            # y: (B,) 或 (B,1) 统一成 (B,)
            y_batch = y_batch.to(device, non_blocking=True).view(-1)

            y_pred = model(X_batch).view(-1)  # (B,)

            # batch MSE
            loss = loss_fn(y_pred, y_batch)
            bsz = y_batch.size(0)
            total_loss += loss.item() * bsz
            total_count += bsz

            # 搬回 CPU
            y_true_np = y_batch.detach().cpu().numpy()
            y_pred_np = y_pred.detach().cpu().numpy()

            di = di_batch.numpy()
            ii = ii_batch.numpy()

            y_true_mat[di, ii] = y_true_np
            y_pred_mat[di, ii] = y_pred_np

            pbar.update(1)

    # ---- 计算 IC：按列（股票）相关，再取均值 ----
    y_true_2d = y_true_mat.reshape(-1, y_true_mat.shape[-1])  # (N, I)
    y_pred_2d = y_pred_mat.reshape(-1, y_pred_mat.shape[-1])  # (N, I)

    iicorr = sapa_rust.nancorr(y_true_2d, y_pred_2d, axis=0, thread=20)
    ic = bn.nanmean(iicorr)

    # ---- 计算 win_rate / 盈亏比 / 平均收益 ----
    y_true_flat = torch.tensor(y_true_mat.reshape(-1), dtype=torch.float32)
    y_pred_flat = torch.tensor(y_pred_mat.reshape(-1), dtype=torch.float32)
    valid_mask = ~torch.isnan(y_true_flat) & ~torch.isnan(y_pred_flat)
    y_true_flat = y_true_flat[valid_mask]
    y_pred_flat = y_pred_flat[valid_mask]

    sign_prod = torch.sign(y_true_flat * y_pred_flat)
    non_zero_mask = y_pred_flat != 0
    win_rate = (y_true_flat * y_pred_flat > 0)[non_zero_mask].float().mean()

    returns = sign_prod * torch.abs(y_true_flat)
    avg_returns = torch.nanmean(returns)

    pos_mask = returns > 0
    neg_mask = returns < 0
    avg_win = returns[pos_mask].mean() if pos_mask.any() else torch.tensor(0.0)
    avg_loss_abs = returns[neg_mask].abs().mean() if neg_mask.any() else torch.tensor(0.0)
    if avg_win == 0 or avg_loss_abs == 0:
        win_loss_ratio = torch.tensor(0.0)
    else:
        win_loss_ratio = avg_win / avg_loss_abs

    avg_loss = total_loss / max(total_count, 1)

    perf_dict = {
        "ic": float(ic),
        "win_rate": float(win_rate),
        "win_loss_ratio": float(win_loss_ratio),
        "avg_returns": float(avg_returns),
        "loss": float(avg_loss),
    }
    return perf_dict


# ========= 训练一个 epoch =========
def train_loop(model, train_loader, val_loader, device, epoch,
               loss_fn, optimizer, log_interval, need_log):

    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (X_batch, y_batch, di_batch, ii_batch) in enumerate(train_loader):
        # X: (B, 60, 158)
        X_batch = X_batch.to(device, non_blocking=True)
        # y: (B,) 或 (B,1) → (B,)
        y_batch = y_batch.to(device, non_blocking=True).view(-1)

        # forward
        y_pred = model(X_batch).view(-1)    # (B,)

        loss = loss_fn(y_pred, y_batch)     # 标量 MSE

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        n_batches += 1

        # step 级别训练 loss
        mlflow.log_metric(
            "train_loss",
            loss_val,
            step=(epoch - 1) * len(train_loader) + batch_idx
        )

        # 按间隔做一次全量 eval 日志
        if need_log and ((batch_idx + 1) % log_interval == 0):
            print(f"\n开始计算第 {batch_idx+1} 步的训练集与验证集 metrics")
            current_step = (epoch - 1) * len(train_loader) + batch_idx

            val_perf = loop_dataloader_get_perf(model, val_loader, device, loss_fn)
            for key, value in val_perf.items():
                mlflow.log_metric(f"val_{key}", float(value), step=current_step)

            train_perf = loop_dataloader_get_perf(model, train_loader, device, loss_fn)
            for key, value in train_perf.items():
                mlflow.log_metric(f"train_{key}", float(value), step=current_step)

            model.train()

        print(
            f"\rEpoch {epoch:02d} Batch {batch_idx}/{len(train_loader)} — loss: {loss_val:.7f}",
            end="", flush=True
        )

    return total_loss / max(n_batches, 1)


# ========= 简单 eval，算 MSE loss =========
def eval_loop(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    print("进入验证集")

    with torch.no_grad():
        for batch_idx, (X_batch, y_batch, di_batch, ii_batch) in enumerate(loader, start=1):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).view(-1)  # (B,)

            y_pred = model(X_batch).view(-1)       # (B,)

            loss = F.mse_loss(y_pred, y_batch)
            total_loss += loss.item()
            n_batches += 1

            print(f"\rbatch {batch_idx}/{len(loader)} — loss: {loss:.4f}", end="", flush=True)

    if n_batches == 0:
        print("\n⚠️ 整个验证集无有效样本，返回 NaN")
        return torch.tensor(float('nan'), device=device)

    return total_loss / n_batches


# ========= 单次训练 run =========
def train_single_run(params: Dict[str, Any], parent_run_id: str = None):
    mlflow.set_tracking_uri(MLFLOW_CONFIG['tracking_uri'])
    mlflow.set_experiment(MLFLOW_CONFIG['experiment_name'])

    dataset_name = params['path_root'].split('/')[-2]
    run_name = (
        f"{params['model_name']}_{params['loss_fn']}_labels_{params['labels_interval']}"
        f"_lr_{params['lr']}_{dataset_name}_{datetime.now().strftime('%m%d')}"
    )
    if params['runname_suffix'] is not None:
        run_name = run_name + params['runname_suffix']

    if parent_run_id:
        with mlflow.start_run(run_name=run_name, nested=True):
            return _train_impl(params)
    else:
        with mlflow.start_run(run_name=run_name):
            return _train_impl(params)


def _train_impl(params: Dict[str, Any]):
    # 记录超参数到 mlflow
    mlflow.log_params(params)

    # 设备
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{int(task.options.gpu)}")
        torch.cuda.set_device(int(task.options.gpu))
        print(f"使用指定GPU: {device}")
    else:
        device = torch.device("cpu")
        print("使用CPU")

    path_root = params['path_root']
    batch_size = params['batch_size']
    num_workers = params['num_workers']
    n_epochs = params['n_epochs']
    log_interval = params['log_interval']
    lr = params['lr']
    universe = params['universe']
    patience = params['patience']
    need_train_log = params['need_train_log']
    seq_len = params.get('seq_len', 60)     # 默认 60
    hidden_dim = params.get('hidden_dim', 128)
    num_layers = params.get('num_layers', 1)
    dropout = params.get('dropout', 0.1)

    if universe == "None":
        universe = None

    train_loader, val_loader, test_loader = get_dataloaders(
        path_root, batch_size, num_workers, universe, seq_len=seq_len
    )

    n_features = train_loader.dataset.n_feat
    print(f"特征数量: {n_features}, 序列长度: {seq_len}, 训练集样本数: {len(train_loader.dataset)}")

    # 模型：AttentionGRU(Res)
    model = AttentionGRURes(
        input_dim=n_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 先 log 一份初始模型（结构）
    mlflow.pytorch.log_model(
        model, name="model",
        registered_model_name="AttentionGRURes"
    )

    # 模型结构 summary
    summ = summary(
        model,
        input_size=(1, seq_len, n_features),
        verbose=0,
        col_names=("input_size", "output_size", "num_params")
    )
    mlflow.log_text(str(summ), "model_summary.txt")

    # 这里把 early stopping 的指标改为 “max IC”
    early_stopping = EarlyStopping(patience=patience, mode="max")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, verbose=True
    )

    # 以 IC 为最佳标准
    best_val_ic = -float('inf')
    best_val_loss_at_best_ic = float('inf')

    save_dir = Path(task.options.path_root) / "nn_ckpt"
    save_dir.mkdir(parents=True, exist_ok=True)

    train_losses = []
    val_losses = []

    loss_fn = nn.MSELoss()

    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoch {epoch:02d} / {n_epochs}")

        # 训练
        train_loss = train_loop(
            model, train_loader, val_loader,
            device, epoch, loss_fn, optimizer,
            log_interval, need_train_log
        )
        train_losses.append(train_loss)

        # 评估 train / val / test metrics（含 IC 等）
        val_perf = loop_dataloader_get_perf(model, val_loader, device, loss_fn)
        val_loss = val_perf['loss']
        val_ic = val_perf['ic']
        val_losses.append(val_loss)

        val_metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in val_perf.items())
        print(f"Val  [{val_metrics_str}]")

        for key, value in val_perf.items():
            v = float(value)
            if math.isnan(v):
                v = 0.0
            mlflow.log_metric(f"epoch_val_{key}", v, step=epoch)

        train_perf = loop_dataloader_get_perf(model, train_loader, device, loss_fn)
        for key, value in train_perf.items():
            v = float(value)
            if math.isnan(v):
                v = 0.0
            mlflow.log_metric(f"epoch_train_{key}", v, step=epoch)

        test_perf = loop_dataloader_get_perf(model, test_loader, device, loss_fn)
        for key, value in test_perf.items():
            v = float(value)
            if math.isnan(v):
                v = 0.0
            mlflow.log_metric(f"epoch_test_{key}", v, step=epoch)

        print(f"Epoch {epoch:02d}: Train Loss = {train_loss:.7f}")

        if optimizer is not None:
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)

        # 如果你想用 IC 做 lr 调度，可以改这里，比如 scheduler.step(-val_ic)
        # 目前先保持注释掉
        # scheduler.step(val_loss)

        # ===== 依据 val_ic 保存当前最优模型 =====
        cur_ic = val_ic if not math.isnan(val_ic) else -1e9
        if cur_ic > best_val_ic:
            best_val_ic = cur_ic
            best_val_loss_at_best_ic = val_loss

            ckpt_name = (
                f"{params['model_name']}_{params['loss_fn']}_labels_{params['labels_interval']}"
                f"_lr_{params['lr']}_epoch{epoch:02d}_valIC{val_ic:.4f}.pt"
            )
            ckpt_path = save_dir / ckpt_name
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model (by val_ic) to {ckpt_path}")
            mlflow.pytorch.log_model(model, artifact_path=f"best_model_epoch{epoch:02d}")

        # ===== 早停也用 val_ic 作为标准 =====
        if early_stopping(cur_ic, model):
            print(f"Early stopping triggered at epoch {epoch} (by val_ic)")
            early_stopping.restore_best_model(model)
            break

    # 最终在 test 上算一个 MSE
    test_loss = eval_loop(model, test_loader, device)
    print(f"\nFinal Test Loss = {test_loss:.4f}")
    mlflow.log_metric("test_loss", float(test_loss), step=len(train_losses) + 1)

    results = {
        'best_val_ic': float(best_val_ic),
        'best_val_loss_at_best_ic': float(best_val_loss_at_best_ic),
        # 向后兼容，如果你之前有用 best_val_loss 这个字段
        'best_val_loss': float(best_val_loss_at_best_ic),
        'test_loss': float(test_loss),
        'final_train_loss': float(train_losses[-1]) if train_losses else float('inf'),
        'final_val_loss': float(val_losses[-1]) if val_losses else float('inf'),
        'epochs_trained': len(train_losses),
    }

    for key, value in results.items():
        mlflow.log_metric(key, value)

    return results


def single_experiment():
    params = {
        'path_root': task.options.path_root,
        'batch_size': task.options.batch_size,
        'num_workers': task.options.num_workers,
        'n_epochs': task.options.n_epochs,
        'log_interval': task.options.log_interval,
        'lr': task.options.lr,
        'alpha': task.options.alpha,
        'patience': task.options.patience,
        'gpu': task.options.gpu,
        'labels_interval': task.options.labels_interval,
        'universe': task.options.universe,
        'loss_fn': task.options.loss_fn,
        'need_train_log': task.options.need_train_log,
        'model_name': task.options.model_name,
        'runname_suffix': task.options.runname_suffix,
        'model': task.options.model,
    }
    return train_single_run(params)


def main():
    print("开始单次实验...")
    single_experiment()


if __name__ == '__main__':
    main()
