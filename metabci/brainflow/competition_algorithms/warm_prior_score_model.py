# -*- coding: utf-8 -*-
"""
Model definition for warm-dominant score regression with imagined weak prior.

This file only builds the model. It does not read data or run experiments.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    n_freqs: int = 100
    n_times: int = 2500
    raw_hidden_dim: int = 48
    task_embed_dim: int = 80
    task_att_hidden: int = 64
    dropout: float = 0.25
    task_mean_residual_ratio: float = 0.20
    prior_lambda: float = 0.04


class RawTaskNodeReadoutBranch(nn.Module):
    """Provided raw branch structure: task_tf [B,K,F,C,T] -> [B,K,3H]."""

    def __init__(self, node_feat_dim, hidden_dim=64, dropout=0.25):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.channel_att = nn.Sequential(
            nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(8, hidden_dim // 2), 1),
        )
        self.out_dim = hidden_dim * 3

    def forward(self, task_tf):
        b, k, freq, c, t = task_tf.shape
        node_x = task_tf.permute(0, 1, 3, 2, 4).contiguous().view(b * k, c, freq * t)
        h = self.node_encoder(node_x)
        score = self.channel_att(h).squeeze(-1)
        weight = torch.softmax(score, dim=1).unsqueeze(-1)
        att_pool = torch.sum(h * weight, dim=1)
        mean_pool = h.mean(dim=1)
        max_pool = h.max(dim=1).values
        emb = torch.cat([att_pool, mean_pool, max_pool], dim=-1)
        return emb.reshape(b, k, -1)


class TaskAttentionPooling(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, dropout=0.25, mean_residual_ratio=0.20):
        super().__init__()
        self.mean_residual_ratio = float(mean_residual_ratio)
        self.att = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, task_emb):
        score = torch.clamp(self.att(task_emb).squeeze(-1), -20.0, 20.0)
        weight = torch.softmax(score, dim=1)
        att_emb = torch.sum(task_emb * weight.unsqueeze(-1), dim=1)
        mean_emb = task_emb.mean(dim=1)
        return att_emb + self.mean_residual_ratio * mean_emb, weight


class WarmPriorScoreRegressor(nn.Module):
    """
    Regress a 1-point subject score.

    warm_tf:  [B,K,F,C,T]
    prior_tf: [B,K,F,C,T]

    imagined prior is weak:
        task_delta = warm_emb - prior_lambda * prior_emb
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.prior_lambda = float(cfg.prior_lambda)
        self.raw_branch = RawTaskNodeReadoutBranch(
            node_feat_dim=cfg.n_freqs * cfg.n_times,
            hidden_dim=cfg.raw_hidden_dim,
            dropout=cfg.dropout,
        )
        self.task_proj = nn.Sequential(
            nn.Linear(self.raw_branch.out_dim, cfg.task_embed_dim),
            nn.LayerNorm(cfg.task_embed_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.task_pool = TaskAttentionPooling(
            cfg.task_embed_dim,
            cfg.task_att_hidden,
            cfg.dropout,
            cfg.task_mean_residual_ratio,
        )
        self.regressor = nn.Sequential(
            nn.Linear(cfg.task_embed_dim, max(16, cfg.task_embed_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(16, cfg.task_embed_dim // 2), 1),
        )

    def forward(self, warm_tf, prior_tf):
        if warm_tf.shape != prior_tf.shape:
            raise ValueError(f"warm_tf and prior_tf must have same shape, got {warm_tf.shape} and {prior_tf.shape}")
        warm_emb = self.raw_branch(warm_tf)
        prior_emb = self.raw_branch(prior_tf)
        task_delta = warm_emb - self.prior_lambda * prior_emb
        task_emb = self.task_proj(task_delta)
        subject_emb, task_weight = self.task_pool(task_emb)
        score = torch.sigmoid(self.regressor(subject_emb)).squeeze(-1)
        return score, {"task_weight": task_weight, "task_emb": task_emb}


def build_model(cfg: Optional[ModelConfig] = None):
    return WarmPriorScoreRegressor(cfg or ModelConfig())
