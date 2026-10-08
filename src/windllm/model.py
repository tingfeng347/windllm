"""Decoder-only Transformer: RMSNorm → RoPE/GQA → SwiGLU."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 1024
    dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2
    hidden_dim: int = 352
    max_seq_len: int = 128
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    def __post_init__(self):
        if (
            min(
                self.vocab_size,
                self.dim,
                self.n_layers,
                self.n_heads,
                self.n_kv_heads,
                self.hidden_dim,
                self.max_seq_len,
            )
            <= 0
        ):
            raise ValueError("模型维度必须为正数")
        if self.dim % self.n_heads or self.n_heads % self.n_kv_heads:
            raise ValueError("dim 必须整除 n_heads，n_heads 必须整除 n_kv_heads")
        if (self.dim // self.n_heads) % 2:
            raise ValueError("RoPE 的 head_dim 必须为偶数")


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # 归一化在 fp32 中计算，避免混合精度下平方溢出。
        y = x.float()
        return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)).to(
            x.dtype
        ) * self.weight


def apply_rope(x, cos, sin):
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.q = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.k = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        repeats = self.n_heads // self.n_kv_heads
        k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
        # SDPA 在支持的 GPU 上自动选择高效实现。右侧 padding 不会影响前面的有效 token。
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).contiguous().view(b, t, -1))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.down = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn = FeedForward(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.ffn(self.ffn_norm(x))


class WindLLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.embedding.weight
        head_dim = cfg.dim // cfg.n_heads
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        angles = torch.outer(torch.arange(cfg.max_seq_len).float(), inv_freq)
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids):
        t = ids.size(1)
        if t > self.config.max_seq_len:
            raise ValueError("输入长度超过 max_seq_len")
        x = self.embedding(ids)
        cos, sin = self.rope_cos[:t].to(x.dtype), self.rope_sin[:t].to(x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.lm_head(self.norm(x))

    @torch.inference_mode()
    def generate(self, ids, max_new_tokens=64, temperature=0.8, top_k=40, eos_id=None):
        if temperature < 0 or top_k < 0 or max_new_tokens < 0:
            raise ValueError("生成参数不能为负数")
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(ids[:, -self.config.max_seq_len :])[:, -1].float()
            if temperature == 0:
                next_id = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                next_id = torch.multinomial(logits.softmax(-1), 1)
            ids = torch.cat((ids, next_id), dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break
        return ids
