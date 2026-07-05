# -*- coding: utf-8 -*-
"""Swallow BCI deep learning models — ported from Fork C (本科生_竞赛1).

Two model architectures based on the 3-modality × 3-branch paradigm:
  - ``ReplacedThreeBranchSwallowNet`` — binary classification (swallow vs rest)
  - ``SwallowQuantificationNet``   — regression (0-100 dysphagia risk score)

Both follow ``metabci.brainda.algorithms.deep_learning`` conventions:
  - ``nn.Module`` subclass with explicit parameter constructors
  - ``cal_backbone()`` returns fused embedding before classifier/regressor head
  - ``_reset_parameters()`` + ``_glorot_weight_zero_bias`` initialisation

References
----------
- ``metabci.brainda.algorithms.deep_learning.base`` — SkorchNet, np_to_th, _glorot_weight_zero_bias
- ``metabci.brainda.algorithms.deep_learning.eegnet`` — EEGNet reference architecture
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Shared utility functions (pure math — copied verbatim)
# ---------------------------------------------------------------------------


def finite_tensor(x: torch.Tensor, nan: float = 0.0,
                  pos: float = 1e4, neg: float = -1e4) -> torch.Tensor:
    """Safe nan_to_num wrapper used across all branches."""
    return torch.nan_to_num(x, nan=nan, posinf=pos, neginf=neg)


def normalize_adj_dense(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Symmetric degree-normalise adjacency matrix."""
    A = torch.clamp(
        torch.nan_to_num(A.float(), nan=0.0, posinf=1.0, neginf=0.0),
        min=0.0, max=1e4)
    deg = A.sum(dim=-1)
    deg_inv_sqrt = torch.pow(deg + eps, -0.5)
    return deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)


def symmetrize_matrix(S: torch.Tensor) -> torch.Tensor:
    """Force matrix symmetry: 0.5 * (S + Sᵀ)."""
    S = finite_tensor(S, nan=0.0, pos=1e4, neg=-1e4)
    return 0.5 * (S + S.transpose(-1, -2))


def fast_spd_trace_normalize(S: torch.Tensor, eps: float = 1e-3,
                              max_abs: float = 1e4) -> torch.Tensor:
    """Trace-normalise SPD matrices."""
    S = symmetrize_matrix(S)
    S = torch.clamp(finite_tensor(S, nan=0.0, pos=max_abs, neg=-max_abs),
                    min=-max_abs, max=max_abs)
    C = S.shape[-1]
    eye = torch.eye(C, device=S.device, dtype=S.dtype).view(
        *([1] * (S.dim() - 2)), C, C)
    S = S + eps * eye
    trace = S.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
    S = S / (trace / C + eps)
    return symmetrize_matrix(S) + eps * eye


def fast_spd_log_vectorize(S: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Log-map SPD matrix to Euclidean vector (upper-tri + log-diag)."""
    S = fast_spd_trace_normalize(S, eps=eps)
    C = S.shape[-1]
    diag = torch.clamp(S.diagonal(dim1=-2, dim2=-1), min=eps, max=1e4)
    scale = torch.sqrt(diag.unsqueeze(-1) * diag.unsqueeze(-2) + eps)
    corr = torch.clamp(S / scale, min=-5.0, max=5.0)
    idx = torch.triu_indices(C, C, offset=0, device=S.device)
    vec = corr[..., idx[0], idx[1]].clone()
    diag_positions = []
    pos = 0
    for i in range(C):
        for j in range(i, C):
            if i == j:
                diag_positions.append(pos)
            pos += 1
    diag_positions = torch.tensor(diag_positions, device=S.device, dtype=torch.long)
    vec[..., diag_positions] = torch.log(diag)
    return torch.clamp(finite_tensor(vec, nan=0.0, pos=12.0, neg=-12.0),
                       min=-12.0, max=12.0)


# ---------------------------------------------------------------------------
# Channel correlation helpers (classification vs quantification variants)
# ---------------------------------------------------------------------------

def channel_corr_cls(x: torch.Tensor, use_abs: bool = True,
                     topk: int = 2, eps: float = 1e-8) -> torch.Tensor:
    """Channel correlation adjacency [B,C,C] — classification variant (topk support)."""
    B, C, T = x.shape
    series = x - x.mean(dim=-1, keepdim=True)
    series = series / (series.std(dim=-1, keepdim=True) + eps)
    A = torch.matmul(series, series.transpose(-1, -2)) / max(T - 1, 1)
    A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A = A.abs() if use_abs else torch.clamp(A, min=0.0)
    if topk is not None and 0 < topk < C:
        A_sparse = torch.zeros_like(A)
        eye_mask = torch.eye(C, device=x.device).bool().view(1, C, C)
        A_no_diag = A.masked_fill(eye_mask, -1e9)
        idx = torch.topk(A_no_diag, k=topk, dim=-1).indices
        A_sparse.scatter_(-1, idx, torch.gather(A, dim=-1, index=idx))
        A = torch.maximum(A_sparse, A_sparse.transpose(-1, -2))
    eye = torch.eye(C, device=x.device, dtype=x.dtype).view(1, C, C)
    return A * (1.0 - eye) + eye


def channel_corr_quant(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Channel correlation adjacency — quantification variant (dense, abs)."""
    B, C, T = x.shape
    if C == 1:
        return torch.ones(B, 1, 1, device=x.device, dtype=x.dtype)
    s = x - x.mean(dim=-1, keepdim=True)
    s = s / (s.std(dim=-1, keepdim=True) + eps)
    A = torch.matmul(s, s.transpose(-1, -2)) / max(T - 1, 1)
    A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0).abs()
    eye = torch.eye(C, device=x.device, dtype=x.dtype).view(1, C, C)
    return A * (1.0 - eye) + eye


# SPD helpers for quantification branch ----------

def construct_spd_nodes_from_raw(x: torch.Tensor, window_size: int = 125,
                                  stride: int = 125, shrinkage: float = 0.05,
                                  eps: float = 1e-4):
    """Sliding-window SPD covariance nodes: [B,C,T] -> ([B,N,C,C], [B,N,1])."""
    B, C, T = x.shape
    nodes, powers = [], []
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        seg = torch.clamp(
            torch.nan_to_num(x[:, :, start:end], nan=0.0, posinf=1e4, neginf=-1e4),
            -20.0, 20.0)
        obs = seg - seg.mean(dim=-1, keepdim=True)
        cov = torch.matmul(obs, obs.transpose(-1, -2)) / max(window_size - 1, 1)
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        eye = torch.eye(C, device=x.device, dtype=x.dtype).view(1, C, C)
        cov = (1.0 - shrinkage) * cov + shrinkage * eye
        cov = cov + eps * eye
        trace = cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
        cov = cov / (trace / C + eps)
        cov = 0.5 * (cov + cov.transpose(-1, -2)) + eps * eye
        power = torch.log(torch.mean(seg ** 2, dim=(1, 2)).clamp_min(eps)).view(B, 1)
        nodes.append(cov)
        powers.append(power)
    if not nodes:
        eye = torch.eye(C, device=x.device, dtype=x.dtype).view(1, C, C).repeat(B, 1, 1)
        nodes = [eye]
        powers = [torch.zeros(B, 1, device=x.device, dtype=x.dtype)]
    return torch.stack(nodes, dim=1), torch.stack(powers, dim=1)


def upper_triangular_vectorize(mat: torch.Tensor) -> torch.Tensor:
    """Vectorise SPD matrix via upper-triangular (no log-diag)."""
    C = mat.shape[-1]
    idx = torch.triu_indices(C, C, offset=0, device=mat.device)
    return mat[..., idx[0], idx[1]]


# ===========================================================================
# Classification model sub-modules
# ===========================================================================


class RawSignalNodeReadoutBranch(nn.Module):
    """Channels-as-nodes with channel-attention MLP."""
    def __init__(self, node_feat_dim: int, hidden_dim: int = 64, dropout: float = 0.20):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.channel_att = nn.Sequential(
            nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(8, hidden_dim // 2), 1))
        self.out_dim = hidden_dim * 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.node_encoder(x)
        score = self.channel_att(h).squeeze(-1)
        weight = torch.softmax(score, dim=1).unsqueeze(-1)
        att_pool = torch.sum(h * weight, dim=1)
        mean_pool = h.mean(dim=1)
        max_pool = h.max(dim=1).values
        return torch.cat([att_pool, mean_pool, max_pool], dim=-1)


class LocalGlobalNodeMLPEncoder(nn.Module):
    """Two-layer MLP with LayerNorm for graph node encoding."""
    def __init__(self, in_dim: int, hidden_dim: int = 64,
                 out_dim: int = 64, dropout: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalGraphFilterLayer(nn.Module):
    """Single-hop graph filter: self + neighbour linear with LayerNorm."""
    def __init__(self, d_model: int, dropout: float = 0.20):
        super().__init__()
        self.lin_self = nn.Linear(d_model, d_model)
        self.lin_neigh = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        A_norm = normalize_adj_dense(A)
        neigh = torch.bmm(A_norm, h)
        out = self.lin_self(h) + self.lin_neigh(neigh)
        out = F.relu(self.norm(out))
        return self.dropout(out)


class LocalGlobalBlock(nn.Module):
    """Local graph filter + global MultiheadAttention + concat/add fusion."""
    def __init__(self, d_model: int = 64, nhead: int = 4, dropout: float = 0.20,
                 use_local: bool = True, use_global: bool = True,
                 use_residual: bool = True, fusion_type: str = "concat"):
        super().__init__()
        self.use_local = bool(use_local)
        self.use_global = bool(use_global)
        self.use_residual = bool(use_residual)
        self.fusion_type = str(fusion_type).lower()
        if d_model % nhead != 0:
            nhead = 1
        self.local_filter = LocalGraphFilterLayer(d_model, dropout) if self.use_local else None
        self.global_attn = (
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            if self.use_global else None)
        if self.fusion_type == "concat":
            parts = 1 + int(self.use_local) + int(self.use_global)
            self.fusion = nn.Sequential(
                nn.Linear(d_model * parts, d_model), nn.LayerNorm(d_model),
                nn.ReLU(), nn.Dropout(dropout))
        else:
            self.fusion = None
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        h = finite_tensor(h)
        A = finite_tensor(A, nan=0.0, pos=1.0, neg=0.0)
        h_local = self.local_filter(h, A) if self.use_local else None
        h_global = None
        if self.use_global:
            h_global, _ = self.global_attn(h, h, h, need_weights=False)
            h_global = finite_tensor(h_global)
        if self.fusion_type == "concat":
            parts = [h]
            if h_local is not None:
                parts.append(h_local)
            if h_global is not None:
                parts.append(h_global)
            h_fuse = self.fusion(torch.cat(parts, dim=-1))
        else:
            h_fuse = torch.zeros_like(h)
            used = 0
            if h_local is not None:
                h_fuse += h_local
                used += 1
            if h_global is not None:
                h_fuse += h_global
                used += 1
            h_fuse = h if used == 0 else h_fuse / float(used)
        return self.out_norm(h + h_fuse) if self.use_residual else self.out_norm(h_fuse)


class ChannelGraphResidualBranchRaw(nn.Module):
    """Build correlation graph from raw time series → LocalGlobalBlock."""
    def __init__(self, node_feat_dim: int, hidden_dim: int = 64,
                 graph_dim: int = 64, dropout: float = 0.20,
                 num_channels: int = 8, topk: int = 2,
                 use_abs_corr: bool = True, nhead: int = 4,
                 use_local: bool = True, use_global: bool = True,
                 use_channel_embedding: bool = True,
                 use_residual: bool = True, fusion_type: str = "concat"):
        super().__init__()
        self.num_channels = num_channels
        self.topk = topk
        self.use_abs_corr = use_abs_corr
        self.use_channel_embedding = use_channel_embedding
        self.node_encoder = LocalGlobalNodeMLPEncoder(
            node_feat_dim, hidden_dim, graph_dim, dropout)
        self.channel_embedding = (
            nn.Embedding(num_channels, graph_dim) if use_channel_embedding else None)
        self.block = LocalGlobalBlock(
            graph_dim, nhead, dropout, use_local, use_global,
            use_residual, fusion_type)
        self.out_dim = graph_dim * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        A = channel_corr_cls(x, use_abs=self.use_abs_corr, topk=self.topk)
        h = self.node_encoder(x)
        if self.use_channel_embedding:
            ids = torch.arange(C, device=x.device)
            h = h + self.channel_embedding(ids).unsqueeze(0)
        h = self.block(h, A)
        mean_pool = h.mean(dim=1)
        max_pool = h.max(dim=1).values
        return torch.cat([mean_pool, max_pool], dim=-1)


class DenseFeatureGraphConv(nn.Module):
    """Simple graph conv: Linear → normalise adj → batch matmul."""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.20):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(B, -1, -1)
        A_norm = normalize_adj_dense(A)
        h = self.lin(x)
        h = torch.bmm(A_norm, h)
        h = F.relu(self.norm(h))
        return self.dropout(h)


class SPDTemporalCovBranchLite(nn.Module):
    """Sliding-window SPD covariance nodes → graph conv → attention pooling."""
    def __init__(self, channels: int, time_len: int = 500,
                 time_window: int = 100, time_stride: int = 100,
                 time_direction: int = 1, shrinkage: float = 0.05,
                 eig_eps: float = 1e-3, hidden_dim: int = 64,
                 graph_dim: int = 64, dropout: float = 0.20,
                 use_power_node_feature: bool = True):
        super().__init__()
        self.channels = channels
        self.time_window = time_window
        self.time_stride = time_stride
        self.shrinkage = shrinkage
        self.eig_eps = eig_eps
        self.use_power_node_feature = use_power_node_feature

        starts = list(range(0, time_len - time_window + 1, time_stride))
        if len(starts) == 0:
            starts = [0]
        self.starts = starts
        N = len(starts)
        A = torch.zeros(N, N, dtype=torch.float32)
        for i in range(N):
            for j in range(N):
                if i == j or abs(i - j) <= time_direction:
                    A[i, j] = 1.0
        self.register_buffer("topology", A)

        spd_vec_dim = channels * (channels + 1) // 2
        node_feat_dim = spd_vec_dim + (1 if use_power_node_feature else 0)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, graph_dim), nn.LayerNorm(graph_dim),
            nn.ReLU(), nn.Dropout(dropout))
        self.graph_conv = DenseFeatureGraphConv(graph_dim, graph_dim, dropout)
        self.node_att = nn.Sequential(
            nn.Linear(graph_dim, max(8, graph_dim // 2)), nn.Tanh(),
            nn.Linear(max(8, graph_dim // 2), 1))
        self.out_dim = graph_dim * 3

    def construct_nodes(self, x: torch.Tensor):
        x = finite_tensor(x, nan=0.0, pos=1e4, neg=-1e4)
        B, C, T = x.shape
        cov_nodes, powers = [], []
        for s in self.starts:
            e = min(s + self.time_window, T)
            seg = x[:, :, s:e]
            obs = seg - seg.mean(dim=-1, keepdim=True)
            obs = torch.clamp(obs, min=-20.0, max=20.0)
            denom = max(obs.shape[-1] - 1, 1)
            cov = torch.matmul(obs, obs.transpose(-1, -2)) / denom
            cov = symmetrize_matrix(cov)
            if self.shrinkage > 0:
                eye_s = torch.eye(C, device=x.device, dtype=x.dtype).view(1, C, C)
                cov = (1.0 - self.shrinkage) * cov + self.shrinkage * eye_s
            cov = fast_spd_trace_normalize(cov, eps=self.eig_eps)
            power_raw = torch.mean(
                torch.clamp(seg, min=-20.0, max=20.0) ** 2,
                dim=(1, 2), keepdim=False).unsqueeze(-1)
            power = torch.log(torch.clamp(power_raw, min=self.eig_eps, max=1e4))
            power = torch.clamp(
                finite_tensor(power, nan=0.0, pos=12.0, neg=-12.0),
                min=-12.0, max=12.0)
            cov_nodes.append(cov)
            powers.append(power)
        return torch.stack(cov_nodes, dim=1), torch.stack(powers, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cov_nodes, power_nodes = self.construct_nodes(x)
        feat = fast_spd_log_vectorize(cov_nodes, eps=self.eig_eps)
        if self.use_power_node_feature:
            feat = torch.cat([feat, power_nodes], dim=-1)
        h = self.node_encoder(feat)
        h = self.graph_conv(h, self.topology.to(device=x.device, dtype=x.dtype))
        score = self.node_att(h).squeeze(-1)
        score = torch.clamp(
            finite_tensor(score, nan=0.0, pos=20.0, neg=-20.0),
            min=-20.0, max=20.0)
        weight = torch.softmax(score, dim=1).unsqueeze(-1)
        att_pool = torch.sum(h * weight, dim=1)
        mean_pool = h.mean(dim=1)
        max_pool = h.max(dim=1).values
        return torch.cat([att_pool, mean_pool, max_pool], dim=-1)


class ProjectionBranchGatedFusion(nn.Module):
    """Raw / Channel / SPD three-branch gated fusion with per-branch projection.

    Decoupled from Config dataclass — accepts explicit parameters.
    """
    def __init__(self, raw_dim: int, ch_dim: int, spd_dim: int,
                 out_dim: int = 96, dropout: float = 0.20,
                 init: tuple = (1.5, 1.0, -0.3),
                 active: tuple = (True, True, True)):
        super().__init__()
        self.raw_proj = nn.Sequential(
            nn.Linear(raw_dim, out_dim), nn.LayerNorm(out_dim),
            nn.ReLU(), nn.Dropout(dropout))
        self.ch_proj = nn.Sequential(
            nn.Linear(ch_dim, out_dim), nn.LayerNorm(out_dim),
            nn.ReLU(), nn.Dropout(dropout))
        self.spd_proj = nn.Sequential(
            nn.Linear(spd_dim, out_dim), nn.LayerNorm(out_dim),
            nn.ReLU(), nn.Dropout(dropout))
        self.branch_logits = nn.Parameter(torch.tensor(init, dtype=torch.float32))
        mask = torch.tensor(active, dtype=torch.bool)
        if not bool(mask.any()):
            raise ValueError("Raw / Channel / SPD: at least one branch required")
        self.register_buffer("active_mask", mask)
        self.out_dim = out_dim

    def forward(self, raw_emb, ch_emb, spd_emb):
        raw = self.raw_proj(finite_tensor(raw_emb))
        ch = self.ch_proj(finite_tensor(ch_emb))
        spd = self.spd_proj(finite_tensor(spd_emb))
        logits = finite_tensor(self.branch_logits, nan=0.0, pos=5.0, neg=-5.0)
        logits = logits.masked_fill(~self.active_mask, -1e4)
        w = torch.softmax(logits, dim=0)
        fused = w[0] * raw + w[1] * ch + w[2] * spd
        return finite_tensor(fused), w


# ===========================================================================
# Classification — ProvidedThreeBranchModalityEncoder (decoupled from cfg)
# ===========================================================================


class ProvidedThreeBranchModalityEncoder(nn.Module):
    """Per-modality encoder: Raw + Channel-Graph + SPD-Cov branches."""
    def __init__(self, in_channels: int, time_len: int,
                 raw_hidden_dim: int = 64,
                 graph_hidden_dim: int = 64, graph_dim: int = 64,
                 spd_hidden_dim: int = 64, spd_graph_dim: int = 64,
                 spd_time_window: int = 100, spd_time_stride: int = 100,
                 spd_time_direction: int = 1,
                 modality_embed_dim: int = 96,
                 dropout: float = 0.20,
                 channel_graph_topk: int = 2,
                 use_abs_corr: bool = True,
                 channel_nhead: int = 4,
                 channel_lg_use_local: bool = True,
                 channel_lg_use_global: bool = True,
                 channel_lg_use_channel_embedding: bool = True,
                 channel_lg_use_residual: bool = True,
                 channel_lg_fusion_type: str = "concat",
                 cov_shrinkage: float = 0.05,
                 spd_eig_eps: float = 1e-3,
                 spd_use_power_node_feature: bool = True,
                 branch_init: tuple = (1.5, 1.0, -0.3),
                 branch_active: tuple = (True, True, True)):
        super().__init__()
        self.use_raw = branch_active[0]
        self.use_channel = branch_active[1]
        self.use_spd = branch_active[2]

        self.raw_branch = RawSignalNodeReadoutBranch(
            time_len, raw_hidden_dim, dropout)
        self.channel_branch = ChannelGraphResidualBranchRaw(
            node_feat_dim=time_len, hidden_dim=graph_hidden_dim,
            graph_dim=graph_dim, dropout=dropout,
            num_channels=in_channels, topk=channel_graph_topk,
            use_abs_corr=use_abs_corr, nhead=channel_nhead,
            use_local=channel_lg_use_local,
            use_global=channel_lg_use_global,
            use_channel_embedding=channel_lg_use_channel_embedding,
            use_residual=channel_lg_use_residual,
            fusion_type=channel_lg_fusion_type)
        self.spd_branch = SPDTemporalCovBranchLite(
            channels=in_channels, time_len=time_len,
            time_window=spd_time_window, time_stride=spd_time_stride,
            time_direction=spd_time_direction,
            shrinkage=cov_shrinkage, eig_eps=spd_eig_eps,
            hidden_dim=spd_hidden_dim, graph_dim=spd_graph_dim,
            dropout=dropout, use_power_node_feature=spd_use_power_node_feature)
        self.fusion = ProjectionBranchGatedFusion(
            raw_dim=self.raw_branch.out_dim,
            ch_dim=self.channel_branch.out_dim,
            spd_dim=self.spd_branch.out_dim,
            out_dim=modality_embed_dim, dropout=dropout,
            init=branch_init, active=branch_active)
        self.out_dim = modality_embed_dim

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        device = x.device
        raw_emb = self.raw_branch(x) if self.use_raw else torch.zeros(
            B, self.raw_branch.out_dim, device=device)
        ch_emb = self.channel_branch(x) if self.use_channel else torch.zeros(
            B, self.channel_branch.out_dim, device=device)
        spd_emb = self.spd_branch(x) if self.use_spd else torch.zeros(
            B, self.spd_branch.out_dim, device=device)
        fused, branch_w = self.fusion(raw_emb, ch_emb, spd_emb)
        return fused, branch_w


# ===========================================================================
# Shared — ModalityGatedFusion (identical in both models)
# ===========================================================================


class ModalityGatedFusion(nn.Module):
    """EEG / EMG / ECG three-modality gated fusion."""
    def __init__(self, embed_dim: int | None = None,
                 init: tuple = (1.0, 1.5, 0.5),
                 active: tuple = (True, True, True)):
        super().__init__()
        self.modality_logits = nn.Parameter(torch.tensor(init, dtype=torch.float32))
        mask = torch.tensor(active, dtype=torch.bool)
        if not bool(mask.any()):
            raise ValueError("EEG / EMG / ECG: at least one modality required")
        self.register_buffer("active_mask", mask)
        self.out_dim = embed_dim

    def forward(self, eeg_emb, emg_emb, ecg_emb):
        logits = finite_tensor(self.modality_logits, nan=0.0, pos=5.0, neg=-5.0)
        logits = logits.masked_fill(~self.active_mask, -1e4)
        w = torch.softmax(logits, dim=0)
        fused = w[0] * eeg_emb + w[1] * emg_emb + w[2] * ecg_emb
        return finite_tensor(fused), w


# ===========================================================================
# Classification — ReplacedThreeBranchSwallowNet
# ===========================================================================


class ReplacedThreeBranchSwallowNet(nn.Module):
    """3-modality × 3-branch swallow classifier.

    Architecture:
      EEG → [Raw/Channel/SPD] → gated fusion → eeg_emb
      EMG → [Raw/Channel/SPD] → gated fusion → emg_emb
      ECG → [Raw/Channel/SPD] → gated fusion → ecg_emb
      ModalityGatedFusion → MLP classifier → logit [B]
    """

    def __init__(self, n_channels_eeg: int = 9, n_channels_emg: int = 6,
                 n_channels_ecg: int = 1, n_samples: int = 500,
                 n_classes: int = 2,
                 raw_hidden_dim: int = 64,
                 graph_hidden_dim: int = 64, graph_dim: int = 64,
                 spd_hidden_dim: int = 64, spd_graph_dim: int = 64,
                 spd_time_window: int = 100, spd_time_stride: int = 100,
                 spd_time_direction: int = 1,
                 modality_embed_dim: int = 96,
                 fusion_hidden_dim: int = 64,
                 dropout: float = 0.20, classifier_dropout: float = 0.30,
                 channel_graph_topk: int = 2,
                 use_abs_corr: bool = True,
                 channel_nhead: int = 4,
                 channel_lg_use_local: bool = True,
                 channel_lg_use_global: bool = True,
                 channel_lg_use_channel_embedding: bool = True,
                 channel_lg_use_residual: bool = True,
                 channel_lg_fusion_type: str = "concat",
                 cov_shrinkage: float = 0.05,
                 spd_eig_eps: float = 1e-3,
                 spd_use_power_node_feature: bool = True,
                 branch_init: tuple = (1.5, 1.0, -0.3),
                 branch_active: tuple = (True, True, True),
                 modality_init: tuple = (1.0, 1.5, 0.5),
                 modality_active: tuple = (True, True, True)):
        super().__init__()
        D = modality_embed_dim
        self._fused_dim = D

        encoder_kw = dict(
            time_len=n_samples, raw_hidden_dim=raw_hidden_dim,
            graph_hidden_dim=graph_hidden_dim, graph_dim=graph_dim,
            spd_hidden_dim=spd_hidden_dim, spd_graph_dim=spd_graph_dim,
            spd_time_window=spd_time_window, spd_time_stride=spd_time_stride,
            spd_time_direction=spd_time_direction,
            modality_embed_dim=D, dropout=dropout,
            channel_graph_topk=channel_graph_topk,
            use_abs_corr=use_abs_corr, channel_nhead=channel_nhead,
            channel_lg_use_local=channel_lg_use_local,
            channel_lg_use_global=channel_lg_use_global,
            channel_lg_use_channel_embedding=channel_lg_use_channel_embedding,
            channel_lg_use_residual=channel_lg_use_residual,
            channel_lg_fusion_type=channel_lg_fusion_type,
            cov_shrinkage=cov_shrinkage, spd_eig_eps=spd_eig_eps,
            spd_use_power_node_feature=spd_use_power_node_feature,
            branch_init=branch_init, branch_active=branch_active)

        self.eeg_encoder = ProvidedThreeBranchModalityEncoder(
            n_channels_eeg, **encoder_kw)
        self.emg_encoder = ProvidedThreeBranchModalityEncoder(
            n_channels_emg, **encoder_kw)
        self.ecg_encoder = ProvidedThreeBranchModalityEncoder(
            n_channels_ecg, **encoder_kw)

        self.modality_fusion = ModalityGatedFusion(
            embed_dim=D, init=modality_init, active=modality_active)

        self.use_eeg = modality_active[0]
        self.use_emg = modality_active[1]
        self.use_ecg = modality_active[2]

        self.classifier = nn.Sequential(
            nn.Linear(D, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim), nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.LayerNorm(fusion_hidden_dim // 2), nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(fusion_hidden_dim // 2, 1))

        self._reset_parameters()

    def _reset_parameters(self):
        """Glorot initialisation — same convention as brainda EEGNet."""
        from metabci.brainda.algorithms.deep_learning.base import (
            _glorot_weight_zero_bias)
        _glorot_weight_zero_bias(self)

    def cal_backbone(self, eeg, emg, ecg):
        """Return fused embedding before classifier head."""
        B = eeg.size(0)
        D = self._fused_dim
        device = eeg.device

        eeg_emb = self.eeg_encoder(eeg)[0] if self.use_eeg else torch.zeros(
            B, D, device=device)
        emg_emb = self.emg_encoder(emg)[0] if self.use_emg else torch.zeros(
            B, D, device=device)
        ecg_emb = self.ecg_encoder(ecg)[0] if self.use_ecg else torch.zeros(
            B, D, device=device)

        fused, _ = self.modality_fusion(eeg_emb, emg_emb, ecg_emb)
        return fused

    def forward(self, eeg, emg, ecg, return_aux: bool = False):
        B = eeg.size(0)
        D = self._fused_dim
        device = eeg.device

        if self.use_eeg:
            eeg_emb, eeg_bw = self.eeg_encoder(eeg)
        else:
            eeg_emb = torch.zeros(B, D, device=device)
            eeg_bw = torch.zeros(3, device=device)
        if self.use_emg:
            emg_emb, emg_bw = self.emg_encoder(emg)
        else:
            emg_emb = torch.zeros(B, D, device=device)
            emg_bw = torch.zeros(3, device=device)
        if self.use_ecg:
            ecg_emb, ecg_bw = self.ecg_encoder(ecg)
        else:
            ecg_emb = torch.zeros(B, D, device=device)
            ecg_bw = torch.zeros(3, device=device)

        fused, modality_w = self.modality_fusion(eeg_emb, emg_emb, ecg_emb)
        logit = self.classifier(fused).squeeze(1)
        logit = torch.nan_to_num(logit, nan=0.0, posinf=20.0, neginf=-20.0)

        if return_aux:
            return logit, {
                "modality_weight": modality_w.detach(),
                "eeg_branch_weight": eeg_bw.detach(),
                "emg_branch_weight": emg_bw.detach(),
                "ecg_branch_weight": ecg_bw.detach()}
        return logit


# ===========================================================================
# Quantification model sub-modules
# ===========================================================================


class ConvGNAct(nn.Module):
    """Conv1d → GroupNorm → GELU → Dropout."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7,
                 groups: int = 8, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        gn_groups = min(groups, out_ch)
        while out_ch % gn_groups != 0 and gn_groups > 1:
            gn_groups -= 1
        self.norm = nn.GroupNorm(gn_groups, out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


class ResidualBlock1D(nn.Module):
    """Two-layer Conv1d residual block."""
    def __init__(self, ch: int, kernel_size: int = 7, dropout: float = 0.2):
        super().__init__()
        self.conv1 = ConvGNAct(ch, ch, kernel_size=kernel_size, dropout=dropout)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=kernel_size,
                               padding=kernel_size // 2, bias=False)
        gn_groups = min(8, ch)
        while ch % gn_groups != 0 and gn_groups > 1:
            gn_groups -= 1
        self.norm = nn.GroupNorm(gn_groups, ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm(self.conv2(h))
        return self.drop(self.act(x + h))


class RawTemporalBranch(nn.Module):
    """Multi-stage 1D conv + residual blocks for raw signal encoding."""
    def __init__(self, in_channels: int, base_channels: int = 32,
                 embed_dim: int = 96, dropout: float = 0.2):
        super().__init__()
        self.stem = ConvGNAct(in_channels, base_channels, kernel_size=15,
                              dropout=dropout)
        self.stage1 = nn.Sequential(
            ResidualBlock1D(base_channels, kernel_size=11, dropout=dropout),
            nn.MaxPool1d(2))
        self.stage2 = nn.Sequential(
            ConvGNAct(base_channels, base_channels * 2, kernel_size=9,
                      dropout=dropout),
            ResidualBlock1D(base_channels * 2, kernel_size=7, dropout=dropout),
            nn.MaxPool1d(2))
        self.stage3 = nn.Sequential(
            ConvGNAct(base_channels * 2, base_channels * 2, kernel_size=5,
                      dropout=dropout),
            ResidualBlock1D(base_channels * 2, kernel_size=5, dropout=dropout))
        self.proj = nn.Sequential(
            nn.Linear(base_channels * 4, embed_dim),
            nn.LayerNorm(embed_dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        avg_pool = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
        max_pool = F.adaptive_max_pool1d(h, 1).squeeze(-1)
        return self.proj(torch.cat([avg_pool, max_pool], dim=-1))


class ChannelGraphBranch(nn.Module):
    """Simpler channel graph: inline local + global + concat fusion."""
    def __init__(self, in_channels: int, time_len: int = 500,
                 hidden_dim: int = 64, embed_dim: int = 96,
                 dropout: float = 0.2, nhead: int = 4):
        super().__init__()
        self.channel_embedding = nn.Embedding(in_channels, hidden_dim)
        self.node_encoder = nn.Sequential(
            nn.Linear(time_len, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU())
        self.lin_self = nn.Linear(hidden_dim, hidden_dim)
        self.lin_neigh = nn.Linear(hidden_dim, hidden_dim)
        self.local_norm = nn.LayerNorm(hidden_dim)
        if hidden_dim % nhead != 0:
            nhead = 1
        self.global_attn = nn.MultiheadAttention(
            hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        A = normalize_adj_dense(channel_corr_quant(x))
        h = self.node_encoder(x)
        channel_ids = torch.arange(C, device=x.device)
        h = h + self.channel_embedding(channel_ids).unsqueeze(0)
        neigh = torch.bmm(A, h)
        h_local = F.gelu(self.local_norm(
            self.lin_self(h) + self.lin_neigh(neigh)))
        h_global, _ = self.global_attn(h, h, h, need_weights=False)
        h_fuse = self.fusion(torch.cat([h, h_local, h_global], dim=-1))
        return self.out_proj(torch.cat(
            [h_fuse.mean(dim=1), h_fuse.max(dim=1).values], dim=-1))


class SPDCovBranch(nn.Module):
    """SPD covariance nodes with attention pooling (no graph conv)."""
    def __init__(self, in_channels: int, embed_dim: int = 96,
                 hidden_dim: int = 64, dropout: float = 0.2,
                 window_size: int = 125, stride: int = 125,
                 shrinkage: float = 0.05, eps: float = 1e-4):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.shrinkage = shrinkage
        self.eps = eps
        cov_vec_dim = in_channels * (in_channels + 1) // 2
        node_feat_dim = cov_vec_dim + 1
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.node_att = nn.Sequential(
            nn.Linear(hidden_dim, max(8, hidden_dim // 2)), nn.Tanh(),
            nn.Linear(max(8, hidden_dim // 2), 1))
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cov_nodes, power_nodes = construct_spd_nodes_from_raw(
            x, self.window_size, self.stride, self.shrinkage, self.eps)
        cov_vec = torch.clamp(
            torch.nan_to_num(upper_triangular_vectorize(cov_nodes),
                             nan=0.0, posinf=12.0, neginf=-12.0),
            -12.0, 12.0)
        h = self.node_encoder(torch.cat([cov_vec, power_nodes], dim=-1))
        score = torch.clamp(
            torch.nan_to_num(self.node_att(h).squeeze(-1),
                             nan=0.0, posinf=20.0, neginf=-20.0),
            -20.0, 20.0)
        w = torch.softmax(score, dim=1).unsqueeze(-1)
        return self.out_proj(torch.cat(
            [torch.sum(h * w, dim=1), h.mean(dim=1),
             h.max(dim=1).values], dim=-1))


class InternalBranchGatedFusion(nn.Module):
    """Simpler branch fusion — weighted sum, no per-branch projection."""
    def __init__(self, init: tuple = (1.2, 1.0, 0.5),
                 active: tuple = (True, True, True)):
        super().__init__()
        self.branch_logits = nn.Parameter(torch.tensor(init, dtype=torch.float32))
        mask = torch.tensor(active, dtype=torch.bool)
        if not bool(mask.any()):
            raise ValueError("Raw / Channel / SPD: at least one branch required")
        self.register_buffer("active_mask", mask)

    def forward(self, raw_emb, ch_emb, spd_emb):
        logits = torch.nan_to_num(
            self.branch_logits, nan=0.0, posinf=5.0, neginf=-5.0)
        logits = logits.masked_fill(~self.active_mask, -1e4)
        w = torch.softmax(logits, dim=0)
        return w[0] * raw_emb + w[1] * ch_emb + w[2] * spd_emb, w


class ModalityEncoder(nn.Module):
    """Per-modality encoder for quantification (RawTemporal + ChannelGraph + SPDCov)."""
    def __init__(self, in_channels: int, time_len: int = 500,
                 embed_dim: int = 96, raw_base_channels: int = 32,
                 graph_hidden_dim: int = 64,
                 spd_hidden_dim: int = 64,
                 spd_window_size: int = 125, spd_stride: int = 125,
                 spd_shrinkage: float = 0.05, spd_eps: float = 1e-4,
                 dropout: float = 0.2,
                 internal_init: tuple = (1.2, 1.0, 0.5),
                 branch_active: tuple = (True, True, True)):
        super().__init__()
        D = embed_dim
        bc = raw_base_channels if in_channels > 1 else max(16, raw_base_channels // 2)
        self.raw_branch = RawTemporalBranch(in_channels, bc, D, dropout)
        self.channel_branch = ChannelGraphBranch(
            in_channels, time_len, graph_hidden_dim, D, dropout)
        self.spd_branch = SPDCovBranch(
            in_channels, D, spd_hidden_dim, dropout,
            spd_window_size, spd_stride, spd_shrinkage, spd_eps)
        self.fusion = InternalBranchGatedFusion(
            init=internal_init, active=branch_active)
        self.out_dim = D

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        D = self.out_dim
        device = x.device
        dtype = x.dtype
        raw_emb = (self.raw_branch(x) if self.fusion.active_mask[0]
                   else torch.zeros(B, D, device=device, dtype=dtype))
        ch_emb = (self.channel_branch(x) if self.fusion.active_mask[1]
                  else torch.zeros(B, D, device=device, dtype=dtype))
        spd_emb = (self.spd_branch(x) if self.fusion.active_mask[2]
                   else torch.zeros(B, D, device=device, dtype=dtype))
        return self.fusion(raw_emb, ch_emb, spd_emb)


# ===========================================================================
# Quantification — SwallowQuantificationNet
# ===========================================================================


class SwallowQuantificationNet(nn.Module):
    """3-modality × 3-branch swallow quantification regressor (0-100 score)."""

    def __init__(self, n_channels_eeg: int = 9, n_channels_emg: int = 6,
                 n_channels_ecg: int = 1, n_samples: int = 500,
                 embed_dim: int = 96, raw_base_channels: int = 32,
                 graph_hidden_dim: int = 64,
                 spd_hidden_dim: int = 64,
                 spd_window_size: int = 125, spd_stride: int = 125,
                 spd_shrinkage: float = 0.05, spd_eps: float = 1e-4,
                 fusion_hidden_dim: int = 64,
                 dropout: float = 0.2,
                 internal_init: tuple = (1.2, 1.0, 0.5),
                 branch_active: tuple = (True, True, True),
                 modality_init: tuple = (1.0, 1.5, 0.5),
                 modality_active: tuple = (True, True, True)):
        super().__init__()
        D = embed_dim
        self._fused_dim = D

        enc_kw = dict(
            time_len=n_samples, embed_dim=D,
            raw_base_channels=raw_base_channels,
            graph_hidden_dim=graph_hidden_dim,
            spd_hidden_dim=spd_hidden_dim,
            spd_window_size=spd_window_size, spd_stride=spd_stride,
            spd_shrinkage=spd_shrinkage, spd_eps=spd_eps,
            dropout=dropout,
            internal_init=internal_init, branch_active=branch_active)

        self.eeg_encoder = ModalityEncoder(n_channels_eeg, **enc_kw)
        self.emg_encoder = ModalityEncoder(n_channels_emg, **enc_kw)
        self.ecg_encoder = ModalityEncoder(n_channels_ecg, **enc_kw)

        self.modality_fusion = ModalityGatedFusion(
            embed_dim=D, init=modality_init, active=modality_active)

        self.use_eeg = modality_active[0]
        self.use_emg = modality_active[1]
        self.use_ecg = modality_active[2]

        self.regressor = nn.Sequential(
            nn.Linear(D, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.LayerNorm(fusion_hidden_dim // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, 1))

        self._reset_parameters()

    def _reset_parameters(self):
        from metabci.brainda.algorithms.deep_learning.base import (
            _glorot_weight_zero_bias)
        _glorot_weight_zero_bias(self)

    def cal_backbone(self, eeg, emg, ecg):
        """Return fused embedding before regressor head."""
        B = eeg.size(0)
        D = self._fused_dim
        device = eeg.device
        dtype = eeg.dtype

        eeg_emb = (self.eeg_encoder(eeg)[0] if self.use_eeg
                   else torch.zeros(B, D, device=device, dtype=dtype))
        emg_emb = (self.emg_encoder(emg)[0] if self.use_emg
                   else torch.zeros(B, D, device=device, dtype=dtype))
        ecg_emb = (self.ecg_encoder(ecg)[0] if self.use_ecg
                   else torch.zeros(B, D, device=device, dtype=dtype))

        fused, _ = self.modality_fusion(eeg_emb, emg_emb, ecg_emb)
        return fused

    def forward(self, eeg, emg, ecg, return_aux: bool = False):
        B = eeg.size(0)
        D = self._fused_dim
        device = eeg.device
        dtype = eeg.dtype

        if self.use_eeg:
            eeg_emb, eeg_bw = self.eeg_encoder(eeg)
        else:
            eeg_emb = torch.zeros(B, D, device=device, dtype=dtype)
            eeg_bw = torch.zeros(3, device=device)
        if self.use_emg:
            emg_emb, emg_bw = self.emg_encoder(emg)
        else:
            emg_emb = torch.zeros(B, D, device=device, dtype=dtype)
            emg_bw = torch.zeros(3, device=device)
        if self.use_ecg:
            ecg_emb, ecg_bw = self.ecg_encoder(ecg)
        else:
            ecg_emb = torch.zeros(B, D, device=device, dtype=dtype)
            ecg_bw = torch.zeros(3, device=device)

        fused, modality_weight = self.modality_fusion(eeg_emb, emg_emb, ecg_emb)
        pred_norm = self.regressor(fused).squeeze(1)
        pred_norm = torch.nan_to_num(pred_norm, nan=0.0, posinf=20.0, neginf=-20.0)

        if return_aux:
            return pred_norm, {
                "modality_weight": modality_weight,
                "eeg_branch_weight": eeg_bw,
                "emg_branch_weight": emg_bw,
                "ecg_branch_weight": ecg_bw}
        return pred_norm
