import torch
import torch.nn as nn

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