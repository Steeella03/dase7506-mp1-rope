"""GPT with rotary attention positions; all other baseline choices are retained.

Hypothesis: relative positions in attention generalize better across the
independent 256-token evaluation windows than a learned absolute embedding.
Compare this model with the unchanged model.py at equal training targets.
"""

import torch
from torch import nn
from torch.nn import functional as F


class RotaryBlock(nn.Module):
    def __init__(self, width, heads, context):
        super().__init__()
        if width % heads or (width // heads) % 2:
            raise ValueError("Each attention head needs an even dimension.")
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

        half_head = width // heads // 2
        positions = torch.arange(context, dtype=torch.float32)
        frequencies = 1.0 / (10000.0 ** (torch.arange(half_head, dtype=torch.float32) / half_head))
        angles = positions[:, None] * frequencies[None, :]
        self.register_buffer("rope_cos", angles.cos()[None, None], persistent=False)
        self.register_buffer("rope_sin", angles.sin()[None, None], persistent=False)

    def rotate(self, x):
        length = x.shape[-2]
        paired = x.reshape(*x.shape[:-1], -1, 2)
        even, odd = paired.unbind(-1)
        cosine, sine = self.rope_cos[:, :, :length], self.rope_sin[:, :, :length]
        return torch.stack((even * cosine - odd * sine,
                            even * sine + odd * cosine), dim=-1).flatten(-2)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(
            self.rotate(q), self.rotate(k), v, is_causal=True
        )
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.norm2(x))


class RotaryGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context = config["context"]
        width = config["width"]
        self.token = nn.Embedding(config["vocab"], width)
        self.blocks = nn.ModuleList([
            RotaryBlock(width, config["heads"], self.context)
            for _ in range(config["depth"])
        ])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config["vocab"], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids):
        x = self.token(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return RotaryGPT(config)
