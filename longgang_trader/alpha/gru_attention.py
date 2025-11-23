import torch
import torch.nn as nn

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
