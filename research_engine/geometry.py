"""Pure geometric representation used by SRM; no benchmark imports."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn


CHANNELS = ("C", "V", "A", "K", "G")


@dataclass(frozen=True)
class GeometryConfig:
    length: int = 22
    neighbors: int = 20
    epsilon: float = 1e-8
    delta: float = 1e-8
    d_model: int = 16
    nhead: int = 2
    num_layers: int = 1
    dim_feedforward: int = 32
    dropout: float = 0.0
    seed: int = 42


def build_curve(rv: np.ndarray, endpoint: int, cfg: GeometryConfig) -> np.ndarray:
    x = np.log(np.maximum(np.asarray(rv, float), cfg.epsilon))
    p = x[endpoint - cfg.length + 1 : endpoint + 1]
    if len(p) != cfg.length:
        raise ValueError("path does not have the specified length")
    z = (p - p.mean()) / (p.std() + cfg.epsilon)
    return np.column_stack((z, np.r_[z[0], z[:-1]], np.r_[z[0], z[0], z[:-2]]))


def descriptors(rv: np.ndarray, endpoint: int, cfg: GeometryConfig) -> dict[str, np.ndarray]:
    c = build_curve(rv, endpoint, cfg)
    v = np.empty_like(c)
    v[1:-1] = (c[2:] - c[:-2]) / 2.0
    v[0], v[-1] = c[1] - c[0], c[-1] - c[-2]
    a = np.empty_like(c)
    a[1:-1] = c[2:] - 2.0 * c[1:-1] + c[:-2]
    a[0], a[-1] = a[1], a[-2]
    k = np.linalg.norm(np.cross(v, a), axis=1) / (np.linalg.norm(v, axis=1) ** 2 + cfg.delta) ** 1.5
    arc = np.linalg.norm(c[1:] - c[:-1], axis=1).sum()
    displacement = np.linalg.norm(c[-1] - c[0])
    g = np.array([arc, displacement, arc / (displacement + cfg.delta)], dtype=float)
    return {"C": c, "V": v, "A": a, "K": k[:, None], "G": g[None, :]}


def query_state(d: dict[str, np.ndarray]) -> np.ndarray:
    tokens = []
    for name in ("C", "V", "A", "K"):
        x = np.asarray(d[name], float).reshape(-1)
        tokens.append((x.mean(), np.sqrt(np.mean(x * x)), x[-1]))
    tokens.append(tuple(np.asarray(d["G"], float).reshape(-1)))
    out = np.asarray(tokens, float)
    if out.shape != (5, 3) or not np.isfinite(out).all():
        raise ValueError("invalid Transformer state")
    return out


class ChannelWeightTransformer(nn.Module):
    """Maps five geometric state tokens to nonnegative channel weights."""
    def __init__(self, cfg: GeometryConfig):
        super().__init__()
        torch.manual_seed(cfg.seed)
        self.channel_embedding = nn.Embedding(5, cfg.d_model)
        self.horizon_embedding = nn.Embedding(3, cfg.d_model)
        self.input_projection = nn.Linear(3, cfg.d_model)
        block = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=cfg.num_layers)
        self.output = nn.Linear(cfg.d_model, 1)

    def forward(self, states: torch.Tensor, horizon: int) -> torch.Tensor:
        if states.ndim != 3 or tuple(states.shape[1:]) != (5, 3):
            raise ValueError("states must have shape [batch, 5, 3]")
        hidx = {1: 0, 5: 1, 21: 2}[int(horizon)]
        batch = states.shape[0]
        channels = torch.arange(5, device=states.device).expand(batch, -1)
        horizons = torch.full((batch, 5), hidx, device=states.device, dtype=torch.long)
        encoded = self.input_projection(states) + self.channel_embedding(channels) + self.horizon_embedding(horizons)
        return torch.softmax(self.output(self.encoder(encoded)).squeeze(-1), dim=-1)
