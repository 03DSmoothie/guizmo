"""Coeur Guizmo : decoder-only RMSNorm + RoPE + GQA + SwiGLU (style Llama3/Qwen3)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import GuizmoConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(-1, keepdim=True)
        x = x / torch.sqrt(var + self.eps)
        return (self.weight * x.to(self.weight.dtype)).type_as(x)


def build_rope_cache(seq_len, head_dim, theta, device, dtype):
    assert head_dim % 2 == 0
    f = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    o = torch.outer(t, f)
    return torch.cos(o).to(dtype), torch.sin(o).to(dtype)


def apply_rope(x, cos, sin):
    L = x.shape[-2]
    cos = cos[:L].unsqueeze(0).unsqueeze(0)
    sin = sin[:L].unsqueeze(0).unsqueeze(0)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class GQAttention(nn.Module):
    def __init__(self, cfg: GuizmoConfig):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.n_kv = cfg.n_kv_head
        self.hd = cfg.n_embd // cfg.n_head
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, self.n_kv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, self.n_kv * self.hd, bias=False)
        self.o_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_head, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv, self.hd).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
            m = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool)).view(1, 1, L, L)
            att = att.masked_fill(~m, float("-inf"))
            y = F.softmax(att, dim=-1) @ v
        y = y.transpose(1, 2).contiguous().view(B, L, self.cfg.n_embd)
        return self.drop(self.o_proj(y))


class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: GuizmoConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.n_embd, cfg.intermediate_size, bias=False)
        self.up = nn.Linear(cfg.n_embd, cfg.intermediate_size, bias=False)
        self.down = nn.Linear(cfg.intermediate_size, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: GuizmoConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd, cfg.rms_norm_eps)
        self.attn = GQAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd, cfg.rms_norm_eps)
        self.mlp = SwiGLUMLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class GuizmoForCausalLM(nn.Module):
    def __init__(self, cfg: GuizmoConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd, padding_idx=cfg.pad_token_id)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, L = idx.shape
        x = self.drop(self.tok_emb(idx))
        cos, sin = build_rope_cache(L, self.cfg.n_embd // self.cfg.n_head, self.cfg.rope_theta, idx.device, torch.float32)
        for blk in self.layers:
            x = blk(x, cos, sin)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=128, temperature=0.8, top_k=50, top_p=0.9, eos_id=2, repetition_penalty=1.1):
        self.eval()
        for _ in range(max_new_tokens):
            cond = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            for b in range(idx.shape[0]):
                for t in set(idx[b].tolist()):
                    if logits[b, t] > 0:
                        logits[b, t] /= repetition_penalty
                    else:
                        logits[b, t] *= repetition_penalty
            if top_k and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            if top_p and top_p < 1.0:
                s_logits, s_idx = torch.sort(logits, descending=True)
                probs = F.softmax(s_logits, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                mask = cum - probs > top_p
                s_logits[mask] = float("-inf")
                logits = torch.full_like(logits, float("-inf")).scatter(1, s_idx, s_logits)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if (nxt == eos_id).all():
                break
        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def load_state_dict_resize(self, other_state: dict, verbose: bool = True):
        """Charge un ckpt meme si vocab a change (435 -> 3742).
        Recopie les lignes communes, init la suite a la moyenne. Re-tie apres."""
        import torch as _t
        own = self.state_dict()
        new = {}
        for k, v in other_state.items():
            if k in own and own[k].shape != v.shape:
                if verbose:
                    print(f"[resize] {k} {tuple(v.shape)} -> {tuple(own[k].shape)}")
                if v.dim() == 2:
                    base = own[k].clone().float()
                    n = min(v.shape[0], own[k].shape[0])
                    base[:n] = v[:n].float()
                    if own[k].shape[0] > v.shape[0]:
                        base[n:] = v.float().mean(dim=0)
                    new[k] = base.to(own[k].dtype)
                else:
                    print(f"[resize] skip {k} (dims differents)")
            else:
                new[k] = v
        miss, unexp = self.load_state_dict(new, strict=False)
        if verbose and (miss or unexp):
            print(f"[resize] missing={miss} unexpected={unexp}")
        if self.cfg.tie_word_embeddings:
            try:
                self.lm_head.weight = self.tok_emb.weight
            except Exception:
                pass
        return miss, unexp


def load_ckpt_compatible(model: "GuizmoForCausalLM", ckpt_path: str, device: str = "cpu"):
    """Utilitaire partage : torch.load + resize vocab auto."""
    import torch as _t
    sd = _t.load(ckpt_path, map_location=device)
    state = sd.get("model", sd) if isinstance(sd, dict) else sd
    if isinstance(sd, dict) and "model" in sd:
        model.load_state_dict_resize(state)
    else:
        model.load_state_dict_resize(state)
    return sd
