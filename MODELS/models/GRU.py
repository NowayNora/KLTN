import torch
import torch.nn as nn
from layers.Embed import DataEmbedding
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

        self.lstm = nn.GRU(
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


























# class ResidualGRUBlock(nn.Module):
#     def __init__(self, dim, bidirectional=False, dropout=0.1):
#         super().__init__()
#         h = dim // 2 if bidirectional else dim
#         self.bigru = nn.GRU(
#             input_size=dim,
#             hidden_size=h,
#             num_layers=1,
#             batch_first=True,
#             bidirectional=bidirectional
#         )
#         out_dim = h * (2 if bidirectional else 1)
#         self.proj = nn.Identity() if out_dim == dim else nn.Linear(out_dim, dim)
#         self.dropout = nn.Dropout(dropout)
#         self.norm = nn.LayerNorm(dim)

#     def forward(self, x):
#         y, _ = self.bigru(x)     # [B,T,out_dim]
#         y = self.proj(y)         # [B,T,dim]
#         y = self.dropout(y)
#         return self.norm(x + y)  # residual


# class AttnPool1Head(nn.Module):
#     """Attention pooling đơn giản (không Transformer)."""
#     def __init__(self, dim):
#         super().__init__()
#         self.q = nn.Parameter(torch.randn(dim))
#         self.proj = nn.Linear(dim, dim)

#     def forward(self, x):
#         k = torch.tanh(self.proj(x))                 # [B,T,dim]
#         s = torch.einsum('btd,d->bt', k, self.q)     # [B,T]
#         w = torch.softmax(s, dim=1).unsqueeze(-1)    # [B,T,1]
#         return (x * w).sum(1)                        # [B,dim]


# class Model(nn.Module):
#     """
#     Deep Residual GRU (no encoder-decoder), mạnh & sâu.
#     Hỗ trợ: long/short forecast, imputation, anomaly_detection, classification.
#     """
#     def __init__(self, configs):
#         super().__init__()
#         self.task_name = configs.task_name
#         self.pred_len  = configs.pred_len
#         self.c_out     = configs.c_out
#         self.seq_len   = getattr(configs, "seq_len", None)

#         d_model   = configs.d_model
#         e_layers  = max(2, configs.e_layers)       # sâu hơn mặc định
#         dropout   = configs.dropout
#         enc_in    = configs.enc_in
#         self.bidirectional = getattr(configs, "bidirectional", True)

#         # ---- Pre-net: project + temporal conv (mở rộng receptive field) ----
#         self.in_proj = nn.Linear(enc_in, d_model)
#         self.temporal_conv = nn.Sequential(
#             nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=1),
#             nn.GELU(),
#             nn.Dropout(dropout)
#         )

#         # ---- Deep Residual GRU stack ----
#         blocks = []
#         for _ in range(e_layers):
#             blocks.append(ResidualGRUBlock(d_model, bidirectional=self.bidirectional, dropout=dropout))
#         self.backbone = nn.Sequential(*blocks)

#         # ---- Readout (multi-pool) ----
#         self.attn_pool = AttnPool1Head(d_model)
#         readout_dim = d_model * 4  # last + mean + max + attn

#         # ---- Heads ----
#         self.forecast_head = nn.Sequential(
#             nn.Linear(readout_dim, d_model * 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model * 2, d_model),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model, self.pred_len * self.c_out)
#         )

#         # per-step head cho imputation/anomaly (giữ chiều theo thời gian)
#         self.per_step_head = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model, self.c_out)
#         )

#         # classification head
#         if self.task_name == 'classification':
#             num_class = configs.num_class
#             self.cls_head = nn.Sequential(
#                 nn.Linear(readout_dim, d_model * 2),
#                 nn.GELU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(d_model * 2, num_class)
#             )

#         # cuối cùng, kích hoạt cho c_out nếu bạn muốn (ví dụ interval in (0,1))
#         self.out_act = {
#             'gelu': F.gelu,
#             'tanh': torch.tanh,
#             'sigmoid': torch.sigmoid,
#             'identity': lambda x: x
#         }.get(getattr(configs, "activation", "identity"), lambda x: x)

#     def _encode(self, x):
#         # x: [B,T,D_in]
#         z = self.in_proj(x)                    # [B,T,d]
#         z = self.temporal_conv(z.transpose(1, 2)).transpose(1, 2)  # Conv1d trên thời gian
#         z = self.backbone(z)                   # [B,T,d]
#         return z

#     def _multipool(self, z):
#         last = z[:, -1, :]
#         mean = z.mean(1)
#         mx   = z.max(1).values
#         attn = self.attn_pool(z)
#         return torch.cat([last, mean, mx, attn], dim=1)  # [B,4d]

#     def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
#         """
#         x_enc: [B, L, D]  (không dùng decoder)
#         """
#         z = self._encode(x_enc)

#         if self.task_name in ['long_term_forecast', 'short_term_forecast']:
#             h = self._multipool(z)                               # [B,4d]
#             y = self.forecast_head(h).view(x_enc.size(0), self.pred_len, self.c_out)
#             return self.out_act(y)

#         if self.task_name in ['imputation', 'anomaly_detection']:
#             y = self.per_step_head(z)                             # [B,L,C]
#             return self.out_act(y)

#         if self.task_name == 'classification':
#             h = self._multipool(z)
#             return self.cls_head(h)                               # [B,num_class]

#         return None
