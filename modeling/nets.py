"""Deep models: LSTM [14], 1D-CNN [8],[15], ANN/MLP [2],[11] and the proposed TGT-M.

TGT-M (Temporal Gated Transformer Model) fuses
  * GRU-D (Che et al., 2018): a gated recurrent encoder with a mask and "time since last
    observation" decay, so the gaps in the seasonal series are modelled, not imputed;
  * Temporal Fusion Transformer (Lim et al., 2021) parts: variable-selection network over the
    static / known-at-t inputs, gated residual networks (GRN), static enrichment and an
    interpretable attention layer with a gated skip connection, plus quantile outputs.
All models receive the same sequence tensor (n, L+1, 10) and the same static vector.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

QUANTILES = (0.1, 0.5, 0.9)
RAIN_SEQ_CH = (7, 8)


class MLP(nn.Module):
    def __init__(self, n_static, hidden=64, drop=0.1, **_):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_static, hidden), nn.ReLU(), nn.Dropout(drop),
                                 nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, seq, st):
        return self.net(st).squeeze(-1)


class StaticMLP(nn.Module):
    def __init__(self, n, h):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(n, h), nn.ReLU())

    def forward(self, x):
        return self.f(x)


class LSTMNet(nn.Module):
    def __init__(self, n_static, hidden=48, drop=0.1, n_ch=10, **_):
        super().__init__()
        self.rnn = nn.LSTM(n_ch, hidden, batch_first=True)
        self.st = StaticMLP(n_static, hidden)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, seq, st):
        _, (h, _) = self.rnn(seq)
        return self.head(torch.cat([h[-1], self.st(st)], -1)).squeeze(-1)


class CNN1D(nn.Module):
    def __init__(self, n_static, hidden=32, drop=0.1, n_ch=10, **_):
        super().__init__()
        self.c1 = nn.Conv1d(n_ch, hidden, 3, padding=1)
        self.c2 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.st = StaticMLP(n_static, hidden)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(3 * hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, seq, st):
        x = seq.transpose(1, 2)
        x = F.relu(self.c2(F.relu(self.c1(x))))
        return self.head(torch.cat([x.mean(-1), x[:, :, -1], self.st(st)], -1)).squeeze(-1)


# ---------------- TGT-M ----------------
class GLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 2 * d)

    def forward(self, x):
        return F.glu(self.fc(x), -1)


class GRN(nn.Module):
    """Gated residual network (TFT): LayerNorm(x + GLU(W2 ELU(W1 x + Wc c)))."""

    def __init__(self, d, d_ctx=None, drop=0.1):
        super().__init__()
        self.w1 = nn.Linear(d, d)
        self.wc = nn.Linear(d_ctx, d, bias=False) if d_ctx else None
        self.w2 = nn.Linear(d, d)
        self.glu = GLU(d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)

    def forward(self, x, c=None):
        h = self.w1(x)
        if self.wc is not None and c is not None:
            h = h + self.wc(c)
        h = self.drop(self.w2(F.elu(h)))
        return self.norm(x + self.glu(h))


class VSN(nn.Module):
    """Variable-selection network over scalar inputs (n_vars); returns context and weights."""

    def __init__(self, n_vars, d, drop=0.1):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_vars, d) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_vars, d))
        self.sel = nn.Sequential(nn.Linear(n_vars, d), nn.ELU(), nn.Linear(d, n_vars))
        self.grn = GRN(d, drop=drop)

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.W + self.b                # (B, V, d)
        w = torch.softmax(self.sel(x), -1)                     # (B, V)
        return self.grn((w.unsqueeze(-1) * emb).sum(1)), w


class GRUD(nn.Module):
    """GRU-D encoder. Channel 0 = relative reading, 1 = mask; other channels are always observed."""

    def __init__(self, n_ch, d):
        super().__init__()
        self.d = d
        self.gx = nn.Linear(1, 1)
        self.gh = nn.Linear(1, d)
        self.cell = nn.GRUCell(n_ch, d)

    def forward(self, seq):
        B, T, _ = seq.shape
        h = seq.new_zeros(B, self.d)
        x_last = seq.new_zeros(B, 1)
        delta = seq.new_zeros(B, 1)
        prev_m = seq.new_ones(B, 1)
        out = []
        for t in range(T):
            x, m = seq[:, t, 0:1], seq[:, t, 1:2]
            delta = torch.where(prev_m > 0, torch.ones_like(delta), delta + 1) if t else delta
            dn = delta / 12.0
            gamma_x = torch.exp(-F.relu(self.gx(dn)))
            x_hat = m * x + (1 - m) * gamma_x * x_last         # decay toward the well's baseline (0)
            gamma_h = torch.exp(-F.relu(self.gh(dn)))
            h = self.cell(torch.cat([x_hat, m, seq[:, t, 2:]], -1), h * gamma_h)
            x_last = torch.where(m > 0, x, x_last)
            prev_m = m
            out.append(h)
        return torch.stack(out, 1)


class TGTM(nn.Module):
    def __init__(self, n_static, hidden=48, drop=0.1, n_ch=10, heads=2, rain_idx=None, **_):
        super().__init__()
        d = hidden
        self.enc = GRUD(n_ch, d)
        self.vsn = VSN(n_static, d, drop)
        self.enrich = GRN(d, d_ctx=d, drop=drop)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.gate = GLU(d)
        self.norm = nn.LayerNorm(d)
        self.pos = GRN(d, drop=drop)
        self.out = nn.Linear(d, 1 + len(QUANTILES))
        self.register_buffer("rain_idx", torch.as_tensor(rain_idx if rain_idx is not None else [], dtype=torch.long))
        self.last_w = None

    def forward(self, seq, st, return_all=False):
        H = self.enc(seq)                                      # (B,T,d)
        c, w = self.vsn(st)
        self.last_w = w.detach()
        E = self.enrich(H, c.unsqueeze(1).expand_as(H))        # static enrichment
        pad = seq[:, :, 6] < 0.5                               # padded steps (before the first round)
        a, aw = self.attn(E[:, -1:], E, E, key_padding_mask=pad, need_weights=True)
        z = self.norm(E[:, -1] + self.gate(a.squeeze(1)))
        o = self.out(self.pos(z))
        self.last_attn = aw.detach()
        return (o[:, 0], o[:, 1:]) if return_all else o[:, 0]


def quantile_loss(q, y):
    loss = 0
    for i, tau in enumerate(QUANTILES):
        e = y - q[:, i]
        loss = loss + torch.maximum(tau * e, (tau - 1) * e).mean()
    return loss / len(QUANTILES)


NETS = {"ANN": MLP, "LSTM": LSTMNet, "CNN1D": CNN1D, "TGT-M": TGTM}


class VSN2(VSN):
    """VSN with a per-variable non-linearity (as in the TFT, each variable gets its own transform)."""

    def forward(self, x):
        emb = F.elu(x.unsqueeze(-1) * self.W + self.b)
        w = torch.softmax(self.sel(x), -1)
        return self.grn((w.unsqueeze(-1) * emb).sum(1)), w


class TGTM2(TGTM):
    """TGT-M v2: per-variable VSN non-linearity and a skip of the selected static context to the output."""

    def __init__(self, n_static, hidden=48, drop=0.1, n_ch=10, heads=2, rain_idx=None, **_):
        super().__init__(n_static, hidden, drop, n_ch, heads, rain_idx)
        self.vsn = VSN2(n_static, hidden, drop)
        self.mix = nn.Linear(2 * hidden, hidden)

    def forward(self, seq, st, return_all=False):
        H = self.enc(seq)
        c, w = self.vsn(st)
        self.last_w = w.detach()
        E = self.enrich(H, c.unsqueeze(1).expand_as(H))
        pad = seq[:, :, 6] < 0.5
        a, aw = self.attn(E[:, -1:], E, E, key_padding_mask=pad, need_weights=True)
        z = self.norm(E[:, -1] + self.gate(a.squeeze(1)))
        z = self.norm(z + F.elu(self.mix(torch.cat([z, c], -1))))
        o = self.out(self.pos(z))
        self.last_attn = aw.detach()
        return (o[:, 0], o[:, 1:]) if return_all else o[:, 0]


NETS["TGT-M-v1"] = TGTM   # first design (VSN weighted sum only); kept for the design-iteration note
NETS["TGT-M"] = TGTM2     # final design
