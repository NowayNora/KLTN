import torch
import torch.nn as nn
from layers.Embed import DataEmbedding
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Vanilla LSTM (bidirectional hoặc không) + dropout
    Input : [B, seq_len, enc_in]
    Output: [B, pred_len, c_out]
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len       = configs.seq_len
        self.pred_len      = configs.pred_len
        self.enc_in        = configs.enc_in
        self.c_out         = configs.c_out
        self.hidden_size   = configs.d_model
        self.num_layers    = configs.e_layers
        self.bidirectional = configs.bidirectional
        self.activation    = F.relu if configs.activation == "relu" else F.gelu
        self.dropout       = configs.dropout

        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout 
        )

        direction_factor = 2 if self.bidirectional else 1
        self.dropout = nn.Dropout(self.dropout)
        self.head = nn.Linear(self.hidden_size * direction_factor,
                              self.pred_len * self.c_out)

    def forward(self, x_enc, *_, **__):
        out, _ = self.lstm(x_enc)          # [B, T, hidden*dir]
        last_h = out[:, -1, :]             # [B, hidden*dir]
        last_h = self.dropout(last_h)      # dropout trên hidden cuối
        pred = self.head(last_h)           # [B, pred_len * c_out]
        pred = pred.view(-1, self.pred_len, self.c_out)
        return self.activation(pred)





























# class ResidualLSTMBlock(nn.Module):
#     def __init__(self, dim, bidirectional=False, dropout=0.1):
#         super().__init__()
#         hidden_per_dir = dim // 2 if bidirectional else dim
#         self.bidirectional = bidirectional
#         self.lstm = nn.LSTM(
#             input_size=dim,
#             hidden_size=hidden_per_dir,
#             num_layers=1,
#             bidirectional=bidirectional,
#             batch_first=True
#         )
#         out_dim = hidden_per_dir * (2 if bidirectional else 1)
#         self.proj = nn.Identity() if out_dim == dim else nn.Linear(out_dim, dim)
#         self.norm = nn.LayerNorm(dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         # x: [B,T,dim]
#         y, _ = self.lstm(x)                 # [B,T,out_dim]
#         y = self.proj(y)                    # [B,T,dim]
#         y = self.dropout(y)
#         y = self.norm(x + y)                # residual
#         return y


# class TemporalAttentionPool(nn.Module):
#     """1-head attention pooling: w = softmax(q · W x_t),  h = Σ w_t x_t"""
#     def __init__(self, dim):
#         super().__init__()
#         self.query = nn.Parameter(torch.randn(dim))  # [dim]
#         self.proj  = nn.Linear(dim, dim)

#     def forward(self, x):
#         # x: [B,T,dim]
#         k = torch.tanh(self.proj(x))                # [B,T,dim]
#         scores = torch.einsum('btd,d->bt', k, self.query)  # [B,T]
#         w = torch.softmax(scores, dim=1).unsqueeze(-1)     # [B,T,1]
#         h = (x * w).sum(dim=1)                      # [B,dim]
#         return h


# class Model(nn.Module):
#     """
#     Deep Residual LSTM (no encoder-decoder) + multi-pool readout (last/mean/max/attn)
#     Output: [B, pred_len, c_out]
#     """
#     def __init__(self, configs):
#         super().__init__()
#         self.pred_len = configs.pred_len
#         self.c_out    = configs.c_out
#         self.d_model  = configs.d_model
#         self.n_blocks = max(2, configs.e_layers)        # sâu hơn
#         self.activation = F.relu if configs.activation == "relu" else F.gelu


#         # ----- Embedding -----
#         self.enc_embedding = DataEmbedding(
#             configs.enc_in, configs.d_model,
#             configs.embed, configs.freq, configs.dropout
#         )

#         # ----- Deep Residual LSTM stack -----
#         blocks = []
#         for _ in range(self.n_blocks):
#             blocks.append(ResidualLSTMBlock(
#                 dim=configs.d_model,
#                 bidirectional=configs.bidirectional,
#                 dropout=configs.dropout
#             ))
#         self.backbone = nn.Sequential(*blocks)

#         # ----- Multi-pooling head -----
#         self.attn_pool = TemporalAttentionPool(configs.d_model)
#         # concat: last + mean + max + attn  => 4 * d_model
#         readout_dim = configs.d_model * 4

#         self.head = nn.Sequential(
#             nn.Linear(readout_dim, configs.d_model * 2),
#             nn.GELU(),
#             nn.Dropout(configs.dropout),
#             nn.Linear(configs.d_model * 2, configs.d_model),
#             nn.GELU(),
#             nn.Dropout(configs.dropout),
#             nn.Linear(configs.d_model, self.pred_len * self.c_out)
#         )

#     def forward(self, x_enc, x_mark_enc, *_, **__):
#         # 1) Embed
#         x = self.enc_embedding(x_enc, x_mark_enc)   # [B,T,d]

#         # 2) Deep residual LSTM
#         x = self.backbone(x)                         # [B,T,d]

#         # 3) Rich readout
#         last = x[:, -1, :]                           # [B,d]
#         mean = x.mean(dim=1)                         # [B,d]
#         mx   = x.max(dim=1).values                   # [B,d]
#         attn = self.attn_pool(x)                     # [B,d]
#         h = torch.cat([last, mean, mx, attn], dim=1) # [B,4d]

#         # 4) Direct multi-step prediction
#         out = self.head(h).view(x.size(0), self.pred_len, self.c_out)
#         return self.activation(out)
