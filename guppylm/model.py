"""
GuppyLM — a tiny fish brain.

Vanilla transformer: multi-head attention, ReLU FFN, LayerNorm, learned positional embeddings.
Optional extensions: Soft MoE FFN, recurrent sublayer, Ouroboros loop.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import GuppyConfig


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(attn, dim=-1))
        return self.out((attn @ v).transpose(1, 2).contiguous().view(B, T, C))


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up = nn.Linear(config.d_model, config.ffn_hidden)
        self.down = nn.Linear(config.ffn_hidden, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.relu(self.up(x))))


class SoftMoE(nn.Module):
    """Soft Mixture of Experts (Puigcerver et al., 2023).

    Replaces the FFN in a transformer block with a pool of expert FFNs.
    Every token participates in every expert via differentiable soft routing —
    no hard top-k gating and no dropped tokens.

    Each expert owns ``moe_slots`` input/output slots.  A shared parameter
    matrix ``phi`` (d_model × n_total_slots) produces per-token logits that
    are normalised in two ways:
      • dispatch weights  — softmax over *tokens*  (how much each token
                            contributes to a given slot)
      • combine  weights  — softmax over *slots*   (how each slot's output
                            is mixed back into a token)

    Expert forward passes are batched via ``torch.bmm`` for parallel
    execution across all experts in a single kernel launch.
    """

    def __init__(self, config: GuppyConfig):
        super().__init__()
        E = config.n_experts
        n_slots = E * config.moe_slots
        self.n_experts = E
        self.slots_per_expert = config.moe_slots

        # Slot embedding matrix — "phi" in the paper
        self.phi = nn.Parameter(torch.empty(config.d_model, n_slots))
        nn.init.normal_(self.phi, std=0.02)

        # Stacked expert weights — shape (E, d_model, ffn_hidden) etc. —
        # allows a single bmm call instead of a sequential Python loop.
        self.w_up   = nn.Parameter(torch.empty(E, config.d_model, config.ffn_hidden))
        self.b_up   = nn.Parameter(torch.zeros(E, config.ffn_hidden))
        self.w_down = nn.Parameter(torch.empty(E, config.ffn_hidden, config.d_model))
        self.b_down = nn.Parameter(torch.zeros(E, config.d_model))
        nn.init.normal_(self.w_up,   std=0.02)
        nn.init.normal_(self.w_down, std=0.02)

        self.dropout = nn.Dropout(config.dropout)

    def _run_experts(self, xs: torch.Tensor) -> torch.Tensor:
        """Apply all experts in parallel via batched matrix multiply.

        Args:
            xs: (B, n_slots, C)
        Returns:
            ys: (B, n_slots, C)
        """
        B, n_slots, C = xs.shape
        S = self.slots_per_expert

        # Reshape to (E, B*S, C) so bmm processes each expert independently
        xs = xs.view(B, self.n_experts, S, C).permute(1, 0, 2, 3).reshape(self.n_experts, B * S, C)

        up   = F.relu(torch.bmm(xs, self.w_up)   + self.b_up.unsqueeze(1))   # (E, B*S, H) — unsqueeze adds slot dim for bmm broadcast
        down =        torch.bmm(up, self.w_down)  + self.b_down.unsqueeze(1)  # (E, B*S, C) — same broadcast pattern

        # Restore (B, n_slots, C)
        return down.view(self.n_experts, B, S, C).permute(1, 0, 2, 3).reshape(B, n_slots, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        logits = x @ self.phi                             # (B, T, n_slots)
        dispatch = torch.softmax(logits, dim=1)           # normalise over tokens
        combine  = torch.softmax(logits, dim=-1)          # normalise over slots

        # Aggregate tokens into slot representations
        xs = torch.einsum("bts,btc->bsc", dispatch, x)   # (B, n_slots, C)

        # Run all experts in parallel
        ys = self._run_experts(xs)                        # (B, n_slots, C)

        # Scatter slot outputs back to token positions
        out = torch.einsum("bts,bsc->btc", combine, ys)  # (B, T, C)
        return self.dropout(out)


class RecurrentLayer(nn.Module):
    """Minimal GRU applied left-to-right across the sequence dimension.

    Processes each position in order, maintaining a hidden state that
    accumulates context from all previous tokens.  The result at each
    position is added as a residual inside the transformer block, giving
    the model an explicit sequential inductive bias alongside attention.

    Note: the token-by-token loop is intentional — the sequential dependency
    is the feature, not a bug.  For short sequences (max_seq_len=128) the
    overhead is negligible; for longer contexts consider a parallel-scan GRU.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.reset  = nn.Linear(d_model * 2, d_model)
        self.update = nn.Linear(d_model * 2, d_model)
        self.new    = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        h = x.new_zeros(B, C)
        outputs: list[torch.Tensor] = []
        for t in range(T):
            xt = x[:, t]                                  # (B, C)
            rz = torch.cat([xt, h], dim=-1)
            r  = torch.sigmoid(self.reset(rz))
            z  = torch.sigmoid(self.update(rz))
            n  = torch.tanh(self.new(torch.cat([xt, r * h], dim=-1)))
            h  = (1 - z) * h + z * n
            outputs.append(h)
        return self.dropout(torch.stack(outputs, dim=1))  # (B, T, C)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = Attention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = SoftMoE(config) if config.use_moe else FFN(config)

        self.use_recurrent = config.use_recurrent
        if config.use_recurrent:
            self.norm3 = nn.LayerNorm(config.d_model)
            self.recurrent = RecurrentLayer(config.d_model, config.dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        if self.use_recurrent:
            x = x + self.recurrent(self.norm3(x))
        return x


class GuppyLM(nn.Module):
    def __init__(self, config: GuppyConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tie weights

        # Ouroboros loop: learned per-iteration offset added before each pass
        if config.use_ouroloop:
            self.loop_emb = nn.Embedding(config.n_loops, config.d_model)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)

        n_loops = self.config.n_loops if self.config.use_ouroloop else 1
        # .weight is accessed on each forward pass because gradients update it every step;
        # unsqueeze(1) adds a per-token broadcast dim: (n_loops, 1, d_model).
        loop_embs = (
            self.loop_emb.weight.unsqueeze(1)   # (n_loops, 1, d_model)
            if self.config.use_ouroloop else None
        )
        for loop_idx in range(n_loops):
            if loop_embs is not None:
                x = x + loop_embs[loop_idx]     # (1, d_model) broadcasts over (B, T, d_model)
            for block in self.blocks:
                x = block(x, mask)

        logits = self.lm_head(self.norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=0,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=64, temperature=0.7, top_k=50, **kwargs):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if next_id.item() == self.config.eos_id:
                break
        return idx, []

    def param_count(self):
        total = sum(p.numel() for p in self.parameters())
        return total, 0

    def param_summary(self):
        total, _ = self.param_count()
        return f"GuppyLM: {total:,} params ({total/1e6:.1f}M)"
