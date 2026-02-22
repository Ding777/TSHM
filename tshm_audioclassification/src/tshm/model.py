# src/tshm/models.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Optional

# -------------------------
# TSHMBlockSimple (unchanged algorithmically; trimmed for clarity)
# -------------------------
class TSHMBlock(nn.Module):
    def __init__(
        self,
        d_model,
        r=32,
        K=8,
        ff_hidden=512,
        gate_kernel=3,
        per_channel_gate=False,
        gate_bias_init=-1.0,
        res_scale_init=0.2,
        causal=False,
    ):
        super().__init__()
        self.d = d_model
        self.causal = bool(causal)
        self.r = int(r)
        self.K = int(K)
        self.gate_kernel = int(gate_kernel)
        self.per_channel_gate = bool(per_channel_gate)

        self.U = nn.Linear(d_model, r, bias=False)
        self.V = nn.Linear(d_model, r, bias=False)
        self.A = nn.Linear(r, K, bias=True)
        self.B = nn.Linear(r, K, bias=True)
        self.c = nn.Parameter(torch.zeros(K))

        out_ch = d_model if per_channel_gate else 1
        self.gate_conv = nn.Conv1d(in_channels=d_model, out_channels=out_ch, kernel_size=self.gate_kernel, padding=0)
        nn.init.normal_(self.gate_conv.weight, std=0.02)
        nn.init.constant_(self.gate_conv.bias, gate_bias_init)

        self.res_scale = nn.Parameter(torch.tensor(float(res_scale_init)))
        self.layer_scale = nn.Parameter(torch.ones(d_model) * 1e-4)

        self.pre_ln = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_hidden),
            nn.GELU() if hasattr(nn, "GELU") else nn.Identity(),
            nn.Linear(ff_hidden, d_model),
        )

        self.register_buffer("_const_w", torch.tensor(0.1))
        self.register_buffer("_const_s", torch.tensor(0.1))

        for lin in (self.U, self.V, self.A, self.B):
            if hasattr(lin, "weight"):
                nn.init.xavier_normal_(lin.weight)

    def forward(self, X):
        B, L, d = X.shape
        assert d == self.d, f"Expected input dim {self.d}, got {d}"
        X_norm = self.pre_ln(X)
        phi = torch.tanh(X_norm)
        P = self.U(phi)
        S = torch.cumsum(P, dim=1)
        Q = self.V(phi)
        eA = self.A(S)
        eB = self.B(Q)
        e = eA + eB + self.c.view(1, 1, -1)
        e = torch.tanh(e)
        G = (self._const_w * (self._const_s ** 2)) * e
        M = torch.einsum("blk,kr->blr", G, self.A.weight)

        if self.causal:
            M_acc = torch.cumsum(M, dim=1)
            term1_pre = torch.einsum("blr,rd->bld", M_acc, self.U.weight)
            N_all = torch.einsum("blk,kr->blr", G, self.B.weight)
            N_acc = torch.cumsum(N_all, dim=1)
            term2_pre = torch.einsum("blr,rd->bld", N_acc, self.V.weight)
            grad = term1_pre + term2_pre
        else:
            M_rev = torch.flip(M, dims=[1])
            M_rev_cumsum = torch.cumsum(M_rev, dim=1)
            M_suf = torch.flip(M_rev_cumsum, dims=[1])
            term1_pre = torch.einsum("blr,rd->bld", M_suf, self.U.weight)
            N_all = torch.einsum("blk,kr->blr", G, self.B.weight)
            term2_pre = torch.einsum("blr,rd->bld", N_all, self.V.weight)
            grad = term1_pre + term2_pre

        g_in = X_norm.permute(0, 2, 1).contiguous()
        if self.causal:
            pad = self.gate_kernel - 1
            g_in_padded = F.pad(g_in, (pad, 0)) if pad > 0 else g_in
            g_raw = self.gate_conv(g_in_padded)
            g_raw = g_raw[..., -L:]
        else:
            pad_total = (self.gate_kernel - 1) // 2
            if pad_total > 0:
                g_in_padded = F.pad(g_in, (pad_total, pad_total))
                g_raw = self.gate_conv(g_in_padded)[..., :L]
            else:
                g_raw = self.gate_conv(g_in)[..., :L]

        g_raw = g_raw.permute(0, 2, 1).contiguous()
        if g_raw.shape[-1] == 1:
            g = torch.sigmoid(g_raw).expand(-1, -1, self.d)
        else:
            g = torch.sigmoid(g_raw)
        eps = 1e-6
        g = eps + (1.0 - eps) * g

        residual = (self.res_scale * g * grad) * self.layer_scale.view(1, 1, -1)
        X_next = X + residual
        X_next = X_next + self.ffn(X_next)
        return X_next

    def init_state(self, batch_size: int, device: torch.device = None):
        if device is None:
            device = next(self.parameters()).device
        B = int(batch_size)
        pad = max(0, self.gate_kernel - 1)
        S = torch.zeros((B, self.r), dtype=torch.float32, device=device)
        M_pref = torch.zeros((B, self.r), dtype=torch.float32, device=device)
        N_pref = torch.zeros((B, self.r), dtype=torch.float32, device=device)
        if pad > 0:
            gate_buf = torch.zeros((B, self.d, pad), dtype=torch.float32, device=device)
        else:
            gate_buf = torch.zeros((B, self.d, 0), dtype=torch.float32, device=device)
        return {"S": S, "M_pref": M_pref, "N_pref": N_pref, "gate_buf": gate_buf}

    def forward_step(self, x_t, state):
        if not self.causal:
            raise RuntimeError("forward_step only supported in causal mode")
        B, d = x_t.shape
        assert d == self.d
        x_norm = self.pre_ln(x_t)
        phi_t = torch.tanh(x_norm)
        phi_prime = 1.0 - phi_t * phi_t
        P_t = self.U(phi_t)
        Q_t = self.V(phi_t)
        state["S"] = state["S"] + P_t
        e_t = self.A(state["S"]) + self.B(Q_t) + self.c.view(1, -1)
        e_t = torch.tanh(e_t)
        G_t = (self._const_w * (self._const_s ** 2)) * e_t
        M_t = torch.einsum("bk,kr->br", G_t, self.A.weight)
        N_t = torch.einsum("bk,kr->br", G_t, self.B.weight)
        state["M_pref"] = state["M_pref"] + M_t
        state["N_pref"] = state["N_pref"] + N_t
        term1 = torch.matmul(state["M_pref"], self.U.weight) * phi_prime
        term2 = torch.matmul(state["N_pref"], self.V.weight) * phi_prime
        grad_t = term1 + term2
        pad = max(0, self.gate_kernel - 1)
        if pad > 0:
            g_in_padded = torch.cat((state["gate_buf"], x_norm.unsqueeze(-1)), dim=-1)
            state["gate_buf"] = g_in_padded[..., 1:].detach()
        else:
            g_in_padded = x_norm.unsqueeze(-1)
        g_conv_out = self.gate_conv(g_in_padded)
        g_raw = g_conv_out.squeeze(-1)
        if g_raw.shape[-1] == 1:
            g = torch.sigmoid(g_raw).expand(-1, self.d)
            g = g.squeeze(-1) if g.dim() == 3 else g
        else:
            if g_raw.shape[-1] == self.d:
                g = torch.sigmoid(g_raw)
            else:
                g = torch.sigmoid(g_raw[:, :1]).expand(-1, self.d)
        eps = 1e-6
        g = eps + (1.0 - eps) * g
        residual = (self.res_scale * g * grad_t) * self.layer_scale.view(1, -1)
        x_next = x_t + residual
        x_next = x_next + self.ffn(x_next)
        return x_next, state


# -------------------------
# Inter-layer helper blocks and stacked wrapper
# -------------------------
class BetweenBlock(nn.Module):
    def __init__(self, d_model, dropout=0.0, activation="gelu", use_layernorm=True, causal=False):
        super().__init__()
        self.use_layernorm = use_layernorm
        self.causal = bool(causal)
        self.layernorm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        if activation == "gelu":
            self.act = nn.GELU() if hasattr(nn, "GELU") else nn.Identity()
        elif activation == "silu":
            self.act = nn.SiLU() if hasattr(nn, "SiLU") else nn.Identity()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
        self.dropout = nn.Dropout(dropout) if (dropout and dropout > 0.0) else nn.Identity()

    def forward(self, x):
        if self.use_layernorm:
            x = self.layernorm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

    def forward_step(self, x_t):
        if self.use_layernorm:
            x_t = self.layernorm(x_t)
        x_t = self.act(x_t)
        x_t = self.dropout(x_t)
        return x_t


class GatedSkip(nn.Module):
    def __init__(self, d_model, init_gate=0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init_gate)))
        self.layer_scale = nn.Parameter(torch.ones(d_model) * 1e-4)

    def forward(self, x_prev, x_curr):
        g = torch.sigmoid(self.alpha)
        return x_curr * g + x_prev * (1.0 - g)

    def forward_step(self, x_prev_t, x_curr_t):
        g = torch.sigmoid(self.alpha)
        return x_curr_t * g + x_prev_t * (1.0 - g)


class TSHMStack(nn.Module):
    def __init__(self, num_layers, d_model, tshm_kwargs=None, between_kwargs=None, use_gated_skip=False, causal=False):
        super().__init__()
        tshm_kwargs = dict(tshm_kwargs or {})
        between_kwargs = dict(between_kwargs or {})
        tshm_kwargs.setdefault("causal", bool(causal))
        between_kwargs.setdefault("causal", bool(causal))
        self.num_layers = int(num_layers)
        self.d_model = int(d_model)
        self.causal = bool(causal)
        self.use_gated_skip = bool(use_gated_skip)
        self.layers = nn.ModuleList([TSHMBlock(d_model=d_model, **tshm_kwargs) for _ in range(self.num_layers)])
        between_defaults = dict(dropout=between_kwargs.get("dropout", 0.0), activation=between_kwargs.get("activation", "gelu"), use_layernorm=between_kwargs.get("use_layernorm", True), causal=between_kwargs.get("causal", False))
        self.betweens = nn.ModuleList([BetweenBlock(d_model=d_model, **between_defaults) for _ in range(max(0, self.num_layers - 1))])
        if self.use_gated_skip:
            self.gates = nn.ModuleList([GatedSkip(d_model=d_model, init_gate=0.0) for _ in range(max(0, self.num_layers - 1))])
        else:
            self.gates = None

    def forward(self, x):
        assert x.dim() == 3 and x.shape[-1] == self.d_model
        x_prev = x
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.betweens):
                x_between = self.betweens[i](x)
                if self.use_gated_skip:
                    x = self.gates[i](x_prev, x_between)
                else:
                    x = x_between
                x_prev = x
        return x

    def init_state(self, batch_size: int, device: torch.device = None):
        if device is None:
            device = next(self.parameters()).device
        state_list = []
        for layer in self.layers:
            s = layer.init_state(batch_size=batch_size, device=device) if hasattr(layer, "init_state") else None
            state_list.append({"tshm": s, "between": None})
        return state_list

    def forward_step(self, x_t, state_list):
        assert self.causal, "forward_step requires causal=True"
        states = state_list
        x_prev_t = x_t
        for i, layer in enumerate(self.layers):
            s = states[i]
            if s is None or s.get("tshm", None) is None:
                raise RuntimeError("Missing per-layer TSHM streaming state; call init_state first")
            x_t, s_tshm = layer.forward_step(x_prev_t, s["tshm"])
            s["tshm"] = s_tshm
            if i < len(self.betweens):
                x_between_t = self.betweens[i].forward_step(x_t)
                if self.use_gated_skip:
                    x_t = self.gates[i].forward_step(x_prev_t, x_between_t)
                else:
                    x_t = x_between_t
                x_prev_t = x_t
        return x_t, states


# -------------------------
# Positional encoding and encoder stack
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        L = x.shape[1]
        return x + self.pe[:, :L, :]


class TSHMEncoder(nn.Module):
    def __init__(self, input_dim, d_model=256, n_layers=3, r=32, K=8, ff_hidden=512, use_pos=True, causal=False):
        super().__init__()
        self.input_dim = input_dim
        self.embed = nn.Linear(input_dim, d_model)
        self.use_pos = use_pos
        self.causal = causal
        if use_pos and not causal:
            self.pos = PositionalEncoding(d_model)
        else:
            self.pos = None
        tshm_kwargs = dict(r=r, K=K, ff_hidden=ff_hidden, gate_kernel=3, per_channel_gate=False, causal=causal)
        between_kwargs = dict(dropout=0.0, activation="gelu", use_layernorm=True, causal=causal)
        self.stack = TSHMStack(num_layers=n_layers, d_model=d_model, tshm_kwargs=tshm_kwargs, between_kwargs=between_kwargs, use_gated_skip=False, causal=causal)
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.embed(x)
        if self.use_pos and self.pos is not None:
            h = self.pos(h)
        h = self.stack(h)
        h = self.out_ln(h)
        return h

    def init_stream_state(self, batch_size: int, device: torch.device = None):
        return self.stack.init_state(batch_size=batch_size, device=device)

    def forward_step(self, x_t, states):
        h = self.embed(x_t)
        h_out, new_states = self.stack.forward_step(h, states)
        h_out = self.out_ln(h_out)
        return h_out, new_states


# -------------------------
# TSHMClassifier (streaming support)
# -------------------------
class TSHMClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        n_classes,
        d_model=256,
        n_layers=3,
        r=32,
        K=8,
        ff_hidden=512,
        use_pos=True,
        dropout=0.2,
        causal=False,
        use_conv=False,
        conv_kernel=3,
        conv_padding_mode: str = "zeros",
        use_residual=True,
        residual_gated=True,
        residual_init_gate: float = 0.0,
        pooling="max",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.n_classes = int(n_classes)
        self.use_conv = bool(use_conv)
        self.use_residual = bool(use_residual)
        self.residual_gated = bool(residual_gated)
        #self.pooling = pooling.lower()
        self.pooling = "max"
        assert self.pooling in ("max", "mean"), "pooling must be 'max' or 'mean'"
        self.causal = bool(causal)

        # pre-conv
        if self.use_conv:
            k = int(conv_kernel)
            if self.causal:
                self.pre_conv = nn.Conv1d(in_channels=self.input_dim, out_channels=self.input_dim, kernel_size=k, stride=1, padding=0, bias=True)
                self._stream_left_pad = k - 1
            else:
                pad = (k - 1) // 2
                self.pre_conv = nn.Conv1d(in_channels=self.input_dim, out_channels=self.input_dim, kernel_size=k, stride=1, padding=pad, bias=True)
                self._stream_left_pad = 0
            nn.init.normal_(self.pre_conv.weight, std=0.02)
            if self.pre_conv.bias is not None:
                nn.init.constant_(self.pre_conv.bias, 0.0)
            self._stream_kernel = k
        else:
            self.pre_conv = None
            self._stream_kernel = 1
            self._stream_left_pad = 0

        self.encoder = TSHMEncoder(input_dim=input_dim, d_model=d_model, n_layers=n_layers, r=r, K=K, ff_hidden=ff_hidden, use_pos=use_pos, causal=causal)

        if self.use_residual:
            self.input_res_proj = nn.Linear(self.input_dim, self.d_model)
            if self.residual_gated:
                self.res_gate_alpha = nn.Parameter(torch.tensor(float(residual_init_gate)))
            else:
                self.res_gate_alpha = None
        else:
            self.input_res_proj = None
            self.res_gate_alpha = None

        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.n_classes),
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 3 and x.shape[-1] == self.input_dim, f"expected (B,L,{self.input_dim})"
        if self.pre_conv is not None:
            x_conv = x.permute(0, 2, 1).contiguous()
            if self.causal and self._stream_left_pad > 0:
                left = self._stream_left_pad
                x_conv = F.pad(x_conv, (left, 0))
                x_conv = self.pre_conv(x_conv)
                x_proc = x_conv.permute(0, 2, 1).contiguous()
            else:
                x_conv = self.pre_conv(x_conv)
                x_proc = x_conv.permute(0, 2, 1).contiguous()
        else:
            x_proc = x
        enc = self.encoder(x_proc)
        if self.use_residual:
            shortcut = self.input_res_proj(x_proc)
            if self.residual_gated and (self.res_gate_alpha is not None):
                gate = torch.sigmoid(self.res_gate_alpha)
                enc = gate * enc + (1.0 - gate) * shortcut
            else:
                enc = enc + shortcut
        pooled = enc.max(dim=1)[0] if self.pooling == "max" else enc.mean(dim=1)
        logits = self.head(pooled)
        return logits

    def init_stream_state(self, batch_size: int, device: Optional[torch.device] = None):
        if device is None:
            device = next(self.parameters()).device
        enc_states = self.encoder.init_stream_state(batch_size=batch_size, device=device)
        if self.pre_conv is not None and self._stream_left_pad > 0:
            B = int(batch_size)
            left = int(self._stream_left_pad)
            conv_buf = torch.zeros((B, self.input_dim, left), dtype=torch.float32, device=device)
            return {"encoder": enc_states, "conv_buf": conv_buf}
        else:
            return enc_states

    def forward_step(self, x_t: torch.Tensor, states):
        if x_t.dim() != 2 or x_t.shape[-1] != self.input_dim:
            raise RuntimeError(f"forward_step expects (B, input_dim) got {tuple(x_t.shape)}")
        if isinstance(states, dict) and "encoder" in states:
            wrapper = True
            encoder_states = states["encoder"]
            conv_buf = states.get("conv_buf", None)
        else:
            wrapper = False
            encoder_states = states
            conv_buf = None

        if self.pre_conv is not None:
            if not wrapper:
                raise RuntimeError("forward_step with pre_conv requires states returned by init_stream_state (wrapper).")
            left = int(self._stream_left_pad)
            k = int(self._stream_kernel)
            if conv_buf is None:
                conv_buf = torch.zeros((x_t.shape[0], self.input_dim, left), dtype=x_t.dtype, device=x_t.device)
            else:
                if conv_buf.device != x_t.device or conv_buf.dtype != x_t.dtype:
                    conv_buf = conv_buf.to(x_t.device).type_as(x_t)
            if left > 0:
                x_col = x_t.unsqueeze(-1)
                if left > 1:
                    conv_buf = torch.cat([conv_buf[:, :, 1:], x_col], dim=2)
                else:
                    conv_buf = x_col
                window = torch.cat([conv_buf, x_col], dim=2)
            else:
                window = x_t.unsqueeze(-1)
            conv_out = F.conv1d(window, self.pre_conv.weight, bias=self.pre_conv.bias, stride=1)
            enc_in = conv_out.squeeze(-1)
            states["conv_buf"] = conv_buf
        else:
            enc_in = x_t

        hidden, new_enc_states = self.encoder.forward_step(enc_in, encoder_states)
        if wrapper:
            states["encoder"] = new_enc_states
            new_states = states
        else:
            new_states = new_enc_states

        if self.use_residual:
            sc_t = self.input_res_proj(enc_in) if self.pre_conv is not None else self.input_res_proj(x_t)
            if self.residual_gated and (self.res_gate_alpha is not None):
                gate = torch.sigmoid(self.res_gate_alpha)
                hidden = gate * hidden + (1.0 - gate) * sc_t
            else:
                hidden = hidden + sc_t

        pooled = hidden
        logits = self.head(pooled)
        # Return hidden explicitly so streaming-eval can pool exactly like batch
        return logits, new_states, hidden
