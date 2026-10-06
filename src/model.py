"""
JEPA Model Architecture.
Core component: JEPAModel.
"""

import os
import hashlib
import time
import contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math
import weakref
from copy import deepcopy
from typing import Dict, Any, List, Optional, Tuple, Sequence


from src.utils import (
    RecipeConfig, 
    TimeEmbedding,
    EMAMomentumUpdater,
    load_schema_csv,
    safe_attention,
    effective_rank
)

# Components

FEATURE_TYPE_PAD = 0
FEATURE_TYPE_CATEGORICAL = 1
FEATURE_TYPE_NUMERIC = 2
PHASE1_ARCHITECTURE_VERSION = "numeric_zset_context_event_jepa_v3"


def _schema_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _require_schema_processed_csv(schema_path: str, expected_sha256: Optional[str] = None) -> str:
    if not schema_path:
        raise ValueError("schema_path is required for type-aware feature encoding.")
    resolved = os.path.abspath(schema_path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Required schema_processed.csv was not found: {resolved}")
    if os.path.basename(resolved) != "schema_processed.csv":
        raise ValueError(f"Expected schema_processed.csv, got: {resolved}")
    actual_sha256 = _schema_sha256(resolved)
    if expected_sha256 and actual_sha256 != str(expected_sha256):
        raise RuntimeError(
            f"schema_processed.csv hash mismatch: expected {expected_sha256}, got {actual_sha256}."
        )
    return resolved


def _load_feature_type_ids(
    schema_path: str,
    num_features: int,
    expected_schema_sha256: Optional[str] = None,
) -> torch.Tensor:
    """
    Load per-VID type ids from schema_processed.csv.
    Type ids:
      0 = pad
      1 = categorical
      2 = numeric
    """
    schema_path = _require_schema_processed_csv(schema_path, expected_schema_sha256)

    schema_df = load_schema_csv(schema_path)
    required = {"vid", "model_type"}
    missing = required - set(schema_df.columns)
    if missing:
        raise ValueError(f"schema_processed.csv missing required columns: {sorted(missing)}")

    table = torch.full((num_features + 1,), FEATURE_TYPE_CATEGORICAL, dtype=torch.long)
    table[0] = FEATURE_TYPE_PAD
    for row in schema_df.itertuples(index=False):
        fid = int(getattr(row, "vid"))
        if fid < 0 or fid > num_features:
            raise ValueError(
                f"Schema VID={fid} exceeds configured num_features={num_features}. "
                "Projector vocabulary and schema must match exactly."
            )
        type_str = str(getattr(row, "model_type", "categorical")).lower()
        if type_str == "numeric":
            table[fid] = FEATURE_TYPE_NUMERIC
        elif type_str == "categorical":
            table[fid] = FEATURE_TYPE_CATEGORICAL
        else:
            raise ValueError(f"Unsupported feature type '{type_str}' for VID={fid} in {schema_path}.")
    return table


def _load_state_feature_id_tables(
    schema_path: str,
    num_features: int,
    expected_schema_sha256: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load state-only VID lists from schema_processed.csv.

    Returns:
      - all state feature ids
      - state numeric feature ids
      - state categorical feature ids
    """
    schema_path = _require_schema_processed_csv(schema_path, expected_schema_sha256)
    schema_df = load_schema_csv(schema_path)
    required = {"vid", "model_type", "feature_category"}
    missing = required - set(schema_df.columns)
    if missing:
        raise ValueError(f"schema_processed.csv missing required columns: {sorted(missing)}")

    state_df = schema_df[schema_df["feature_category"].astype(str).str.lower() == "state"].copy()
    vids = state_df["vid"].astype(int).tolist()
    for fid in vids:
        if fid < 0 or fid > num_features:
            raise ValueError(
                f"Schema VID={fid} exceeds configured num_features={num_features}. "
                "Projector vocabulary and schema must match exactly."
            )
    num_df = state_df[state_df["model_type"].astype(str).str.lower() == "numeric"]
    cat_df = state_df[state_df["model_type"].astype(str).str.lower() == "categorical"]
    return (
        torch.tensor(sorted(set(vids)), dtype=torch.long),
        torch.tensor(sorted(set(num_df["vid"].astype(int).tolist())), dtype=torch.long),
        torch.tensor(sorted(set(cat_df["vid"].astype(int).tolist())), dtype=torch.long),
    )


def _safe_corr_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.numel() < 2 or y.numel() < 2:
        return torch.tensor(0.0, device=x.device)
    x = x.float().view(-1)
    y = y.float().view(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x.square().sum()) * (y.square().sum())).clamp(min=1e-8)
    return (x * y).sum() / denom


def _ridge_readout_stats(z: torch.Tensor, y: torch.Tensor, ridge: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Lightweight in-batch linear recoverability diagnostic.
    Returns:
      r2, corr(y_hat, y)
    """
    if z.ndim != 2:
        z = z.view(z.size(0), -1)
    y = y.float().view(-1, 1)
    if z.size(0) < 4 or y.size(0) != z.size(0):
        zero = torch.tensor(0.0, device=z.device)
        return zero, zero

    # Validation may run under autocast/bfloat16. Keep this diagnostic readout in
    # explicit float32 so linear algebra stays stable and dtype-consistent.
    with torch.autocast(device_type=z.device.type, enabled=False):
        x = z.detach().to(torch.float32)
        y = y.detach().to(torch.float32)
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=torch.float32)
        x_aug = torch.cat([x, ones], dim=1)
        xtx = x_aug.T @ x_aug
        eye = torch.eye(xtx.size(0), device=x.device, dtype=torch.float32)
        eye[-1, -1] = 0.0  # do not regularize intercept
        xty = x_aug.T @ y
        w = torch.linalg.solve(xtx + ridge * eye, xty)
        y_hat = x_aug @ w

    ss_res = (y - y_hat).square().sum()
    ss_tot = (y - y.mean()).square().sum()
    if ss_tot <= 1e-8:
        r2 = torch.tensor(0.0, device=x.device)
    else:
        r2 = 1.0 - (ss_res / ss_tot)
    corr = _safe_corr_1d(y_hat.squeeze(-1), y.squeeze(-1))
    return r2, corr




class StabilityClamp(nn.Module):
    """
    Small bounded nonlinearity used by selected heads to prevent runaway outputs
    while staying close to identity near zero.
    """
    def __init__(self, limit: float = 10.0):
        super().__init__()
        self.limit = limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.limit * torch.tanh(x / self.limit)


class FeatureProjector(nn.Module):
    """
    Unified type-aware token encoder for states and interventions.

    Numeric and categorical tokens are intentionally encoded differently:
      - categorical: identity + presence semantics
      - numeric:     identity + standardized severity embedding

    This preserves EHR-specific value information without forcing all tokens
    through the same scalar-broadcast modulation path.
    """
    def __init__(self, embed_dim: int, num_features: int, feature_type_ids: torch.Tensor):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_features = num_features
        if feature_type_ids.numel() != num_features + 1:
            raise ValueError(
                f"feature_type_ids must have shape ({num_features + 1},), "
                f"got {tuple(feature_type_ids.shape)}"
            )
        self.register_buffer("feature_type_ids", feature_type_ids.long().clone(), persistent=True)

        # Identity semantics
        self.item_embedding = nn.Embedding(
            num_features + 1, embed_dim, padding_idx=0)
        nn.init.trunc_normal_(self.item_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

        # Type semantics
        self.type_embedding = nn.Embedding(3, embed_dim, padding_idx=FEATURE_TYPE_PAD)
        nn.init.trunc_normal_(self.type_embedding.weight, std=0.02)
        with torch.no_grad():
            self.type_embedding.weight[FEATURE_TYPE_PAD].zero_()

        # Categorical / presence path
        self.presence_bias = nn.Embedding(num_features + 1, embed_dim, padding_idx=0)
        nn.init.zeros_(self.presence_bias.weight)
        with torch.no_grad():
            self.presence_bias.weight[0].zero_()

        # Numeric / severity path
        self.numeric_gate = nn.Embedding(num_features + 1, embed_dim, padding_idx=0)
        nn.init.zeros_(self.numeric_gate.weight)
        with torch.no_grad():
            self.numeric_gate.weight[0].zero_()
        self.numeric_bias = nn.Embedding(num_features + 1, embed_dim, padding_idx=0)
        nn.init.zeros_(self.numeric_bias.weight)
        with torch.no_grad():
            self.numeric_bias.weight[0].zero_()
        self.value_mlp = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.output_norm = nn.LayerNorm(embed_dim)

    def encode_identity(self, code: torch.Tensor) -> torch.Tensor:
        """Return the shared VID identity embedding used by JEPA queries."""
        return self.item_embedding(code)

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, code: torch.Tensor, val: torch.Tensor,
                var_dummy: Any = None) -> torch.Tensor:
        """
        Args:
            code: (..., S) Long  — Feature / Intervention IDs.
            val:  (..., S, 1) Float — Values (1.0 for categorical).
        Returns:
            z: (..., S, D) — type-aware token embeddings (zero for PAD).
        """
        type_ids = self.feature_type_ids[code]
        z_id = self.item_embedding(code)
        z_type = self.type_embedding(type_ids)

        # Presence branch: categorical tokens are primarily identity + occurrence.
        z_cat = z_id + z_type + self.presence_bias(code)

        # Severity branch: numeric tokens receive a dedicated value embedding.
        val_scalar = val if val.size(-1) == 1 else val[..., :1]
        val_feats = torch.cat(
            [val_scalar, val_scalar.square(), torch.sign(val_scalar)],
            dim=-1
        )
        z_val = self.value_mlp(val_feats)
        gate = torch.sigmoid(self.numeric_gate(code))
        z_num = z_id + z_type + self.numeric_bias(code) + gate * z_val

        numeric_mask = (type_ids == FEATURE_TYPE_NUMERIC).unsqueeze(-1)
        z = torch.where(numeric_mask, z_num, z_cat)
        z = self.output_norm(z)

        pad_mask = (code == 0).unsqueeze(-1)          # (..., S, 1)
        z = z.masked_fill(pad_mask, 0.0)

        return z

class _SetAttentionBlock(nn.Module):
    """
    Lightweight pre-norm self-attention block for intra-visit token mixing.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, set_len, dim = x.shape
        x_norm = self.norm1(x)

        q = self.q_proj(x_norm).view(bsz, set_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(bsz, set_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(bsz, set_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = safe_attention(
            q, k, v, mask=mask.view(bsz, 1, 1, set_len),
            dropout_p=self.drop.p if self.training else 0.0,
            return_attn=False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, set_len, dim)
        x = x + self.drop(self.out_proj(attn_out))
        x = x.masked_fill(mask.unsqueeze(-1), 0.0)

        x = x + self.ff(self.norm2(x))
        x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        return x


class SetEncoder(nn.Module):
    """
    Intra-visit state-token encoder:
      1) Lightweight self-attention token mixer (pre-norm blocks)
      2) Learned gated pooling
      3) Auxiliary masked max pooling
      4) Fusion projection to visit state (B*T, 1, D)
    """
    def __init__(self, config: RecipeConfig, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        depth = max(1, int(getattr(config, "set_layers", 2)))

        self.token_mixer = nn.ModuleList([
            _SetAttentionBlock(embed_dim, num_heads, dropout=0.1)
            for _ in range(depth)
        ])

        self.gate_proj = nn.Linear(embed_dim, embed_dim)
        self.gate_score = nn.Linear(embed_dim, 1, bias=False)
        self.fuse = nn.Linear(embed_dim * 2, embed_dim)
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        return_attn: bool = False,
        return_tokens: bool = False,
    ):
        """
        x:    (B*T, S, D) token embeddings
        mask: (B*T, S)    bool, True=invalid/padding/masked token
        return_attn: if True, include gated-pooling weights ``alpha``.
        return_tokens: if True, include post-token-mixer states ``h``.

        Return contract:
          default                         -> pooled
          return_attn=True                -> pooled, alpha
          return_tokens=True              -> pooled, h
          return_attn=True, return_tokens -> pooled, h, alpha
        """
        if mask is None:
            mask = torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
        else:
            mask = mask.bool()

        h = x
        for block in self.token_mixer:
            h = block(h, mask)
        h = h.masked_fill(mask.unsqueeze(-1), 0.0)

        valid = ~mask
        all_masked = (~valid).all(dim=1, keepdim=True)  # (B*T, 1)

        # Learned gated pooling.
        scores = self.gate_score(torch.tanh(self.gate_proj(h))).squeeze(-1)  # (B*T, S)
        scores = scores.masked_fill(mask, float("-inf"))
        scores = scores.masked_fill(all_masked.expand_as(scores), 0.0)
        alpha = torch.softmax(scores, dim=-1)
        alpha = alpha.masked_fill(mask, 0.0)
        alpha = alpha.masked_fill(all_masked.expand_as(alpha), 0.0)
        z_attn = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)  # (B*T, D)

        # Auxiliary extremum branch to preserve sparse critical signals.
        h_for_max = h.masked_fill(mask.unsqueeze(-1), float("-inf"))
        z_max = h_for_max.max(dim=1).values
        z_max = torch.nan_to_num(z_max, nan=0.0, posinf=0.0, neginf=0.0)
        z_max = z_max.masked_fill(all_masked.expand(-1, z_max.size(-1)), 0.0)

        out = self.fuse(torch.cat([z_attn, z_max], dim=-1))
        out = self.final_norm(out)
        out = out.masked_fill(all_masked.expand(-1, out.size(-1)), 0.0)
        pooled = out.unsqueeze(1)
        if return_attn and return_tokens:
            return pooled, h, alpha
        if return_tokens:
            return pooled, h
        if return_attn:
            return pooled, alpha
        return pooled


class ZConstructor(nn.Module):
    """Value-blind masked-VID query encoder sharing FeatureProjector identity."""

    def __init__(self, embed_dim: int, feature_projector: FeatureProjector):
        super().__init__()
        self.embed_dim = int(embed_dim)
        # Weak reference deliberately avoids registering a second module alias
        # in the checkpoint while preserving the shared identity embedding.
        self._feature_projector_ref = weakref.ref(feature_projector)
        self.query_phi = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    @property
    def feature_projector(self) -> FeatureProjector:
        projector = self._feature_projector_ref()
        if projector is None:
            raise RuntimeError("Shared FeatureProjector is no longer available")
        return projector

    def forward(
        self,
        codes: torch.Tensor,
        is_masked: torch.Tensor,
        token_pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build one query for every masked numeric token from its shared VID ID."""
        if codes.shape != is_masked.shape:
            raise ValueError("codes and is_masked must have identical (B,T,S) shapes")
        query_pad_mask = ~is_masked.bool()
        if token_pad_mask is not None:
            query_pad_mask = query_pad_mask | token_pad_mask.bool()
        z_id = self.feature_projector.encode_identity(codes)
        query = z_id + self.query_phi(z_id)
        return query.masked_fill(query_pad_mask.unsqueeze(-1), 0.0), query_pad_mask


class RelativeTimeTransformerLayer(nn.Module):
    """Phase-1 temporal self-attention with signed pairwise relative time."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.relative_time_embedding = TimeEmbedding(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def _relative_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query_times: torch.Tensor,
        key_times: torch.Tensor,
        key_pad_mask: torch.Tensor,
        signed_log_dt: Optional[torch.Tensor] = None,
        additive_key_mask: Optional[torch.Tensor] = None,
        disable_relative_time_bias: bool = False,
    ) -> torch.Tensor:
        """Exact relative-time attention without materializing all query/key embeddings."""
        B, _, Q, _ = q.shape
        K = k.size(2)
        chunk = max(1, int(getattr(self, "relative_query_chunk_size", 64)))
        attn_mask = additive_key_mask
        if attn_mask is None:
            attn_mask = torch.zeros(B, 1, 1, K, dtype=q.dtype, device=q.device)
            attn_mask = attn_mask.masked_fill(key_pad_mask.view(B, 1, 1, K), float("-inf"))
        chunks: List[torch.Tensor] = []
        for start in range(0, Q, chunk):
            end = min(Q, start + chunk)
            q_chunk = q[:, :, start:end]
            if disable_relative_time_bias:
                chunks.append(safe_attention(
                    q_chunk, k, v, mask=attn_mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                ))
                continue
            if signed_log_dt is None:
                signed_dt = key_times.unsqueeze(1) - query_times[:, start:end].unsqueeze(2)
                signed_log_dt_chunk = torch.sign(signed_dt) * torch.log1p(signed_dt.abs())
            else:
                signed_log_dt_chunk = signed_log_dt[:, start:end]
            rel_chunk = self.relative_time_embedding(
                signed_log_dt_chunk.unsqueeze(-1)
            ).view(B, end - start, K, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
            score_bias_chunk = (q_chunk.unsqueeze(3) * rel_chunk).sum(dim=-1) / math.sqrt(self.head_dim)
            chunks.append(safe_attention(
                q_chunk, k, v, mask=attn_mask, score_bias=score_bias_chunk,
                dropout_p=self.dropout.p if self.training else 0.0,
            ))
        return torch.cat(chunks, dim=2)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        times: torch.Tensor,
        signed_log_dt: Optional[torch.Tensor] = None,
        additive_key_mask: Optional[torch.Tensor] = None,
        disable_relative_time_bias: bool = False,
    ) -> torch.Tensor:
        B, T, D = x.shape
        time_2d = times.squeeze(-1) if times.dim() == 3 else times
        if time_2d.shape != (B, T):
            raise ValueError(f"Expected times (B,T), got {tuple(time_2d.shape)}")
        h = self.norm1(x)
        q = self.q_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        out = self._relative_attention(
            q, k, v, time_2d, time_2d, pad_mask,
            signed_log_dt=signed_log_dt,
            additive_key_mask=additive_key_mask,
            disable_relative_time_bias=disable_relative_time_bias,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.dropout(self.out_proj(out))
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        x = x + self.ff(self.norm2(x))
        return x.masked_fill(pad_mask.unsqueeze(-1), 0.0)


class RelativeTimeTransformerEncoder(nn.Module):
    """Phase-1-only temporal backbone; never changes generic encoder semantics."""

    def __init__(self, embed_dim: int, num_heads: int, layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            RelativeTimeTransformerLayer(embed_dim, num_heads, dropout)
            for _ in range(max(1, int(layers)))
        ])
        self.activation_checkpointing = False

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor, times: torch.Tensor, causal_mask=None) -> torch.Tensor:
        if causal_mask is not None:
            raise ValueError("RelativeTimeTransformerEncoder is bidirectional-only for Phase 1")
        time_2d = times.squeeze(-1) if times.dim() == 3 else times
        signed_dt = time_2d.unsqueeze(1) - time_2d.unsqueeze(2)
        signed_log_dt = torch.sign(signed_dt) * torch.log1p(signed_dt.abs())
        additive_key_mask = torch.zeros(
            x.size(0), 1, 1, x.size(1), dtype=x.dtype, device=x.device
        ).masked_fill(pad_mask.view(x.size(0), 1, 1, x.size(1)), float("-inf"))
        h = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        for layer in self.layers:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(
                    layer, h, pad_mask, times, signed_log_dt, additive_key_mask, use_reentrant=False
                )
            else:
                h = layer(h, pad_mask, times, signed_log_dt, additive_key_mask)
        return h.masked_fill(pad_mask.unsqueeze(-1), 0.0)


class RelativeTimeCrossAttentionLayer(nn.Module):
    """Query-to-temporal-context attention using signed query/key time gaps."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_ff = nn.LayerNorm(embed_dim)
        self.relative_time_embedding = TimeEmbedding(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        query_times: torch.Tensor,
        kv_times: torch.Tensor,
        kv_pad_mask: torch.Tensor,
        query_pad_mask: Optional[torch.Tensor] = None,
        signed_log_dt: Optional[torch.Tensor] = None,
        additive_key_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Q, D = query.shape
        K = kv.size(1)
        q = self.q_proj(self.norm_q(query)).view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        kvn = self.norm_kv(kv)
        k = self.k_proj(kvn).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kvn).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        chunk = max(1, int(getattr(self, "relative_query_chunk_size", 64)))
        attn_mask = additive_key_mask
        if attn_mask is None:
            attn_mask = torch.zeros(B, 1, 1, K, dtype=q.dtype, device=query.device)
            attn_mask = attn_mask.masked_fill(kv_pad_mask.view(B, 1, 1, K), float("-inf"))
        chunks: List[torch.Tensor] = []
        for start in range(0, Q, chunk):
            end = min(Q, start + chunk)
            q_chunk = q[:, :, start:end]
            if signed_log_dt is None:
                signed_dt = kv_times.unsqueeze(1) - query_times[:, start:end].unsqueeze(2)
                signed_log_dt_chunk = torch.sign(signed_dt) * torch.log1p(signed_dt.abs())
            else:
                signed_log_dt_chunk = signed_log_dt[:, start:end]
            rel_chunk = self.relative_time_embedding(
                signed_log_dt_chunk.unsqueeze(-1)
            ).view(B, end - start, K, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
            score_bias_chunk = (q_chunk.unsqueeze(3) * rel_chunk).sum(dim=-1) / math.sqrt(self.head_dim)
            chunks.append(safe_attention(
                q_chunk, k, v, mask=attn_mask, score_bias=score_bias_chunk,
                dropout_p=self.dropout.p if self.training else 0.0,
            ))
        out = torch.cat(chunks, dim=2)
        out = out.transpose(1, 2).contiguous().view(B, Q, D)
        h = query + self.dropout(self.out_proj(out))
        h = h + self.ff(self.norm_ff(h))
        if query_pad_mask is not None:
            h = h.masked_fill(query_pad_mask.unsqueeze(-1), 0.0)
        return h


class JEPAPredictor(nn.Module):
    """
    Phase 1 predictor with structured query attention.

    Design:
    1) Each query is fused with its partial local z_set at the same timestamp.
    2) Each query then cross-attends to the full temporal context states.

    Masked queries never attend to one another.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        layers: int,
        dropout: float = 0.1,
        use_clamp: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.local_fusion = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.context_cross_layers = nn.ModuleList([
            RelativeTimeCrossAttentionLayer(embed_dim, num_heads, dropout) for _ in range(layers)
        ])
        self.activation_checkpointing = False
        
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        
        # Industrial Hardening boundary
        self.use_clamp = use_clamp
        self.clamp = StabilityClamp(limit=10.0) if use_clamp else nn.Identity()
        
    def forward(
        self,
        s_context: torch.Tensor,
        z_query_tokens: torch.Tensor,
        z_set: torch.Tensor,
        state_times: torch.Tensor,
        context_pad_mask: Optional[torch.Tensor] = None,
        query_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            s_context: (B, T, D) temporal context
            z_query_tokens: (B, T, S, D)
            z_set: (B, T, D) partial local state summaries
            state_times: (B, T[,1])
            context_pad_mask: (B, T) bool
            query_pad_mask: (B, T, S) bool

        Returns:
            preds: (B, T, S, D)
        """
        B, T, D = s_context.shape
        _, T_q, S_q, D_q = z_query_tokens.shape
        if T_q != T:
            raise ValueError(f"query timestep count ({T_q}) must match context timestep count ({T})")
        if D_q != D:
            raise ValueError(f"query dim ({D_q}) must match context dim ({D})")

        if context_pad_mask is None:
            context_pad_mask = torch.zeros(B, T, dtype=torch.bool, device=s_context.device)
        if query_pad_mask is None:
            query_pad_mask = torch.zeros(B, T, S_q, dtype=torch.bool, device=s_context.device)

        if z_set.shape != (B, T, D):
            raise ValueError("z_set must have shape (B, T, D)")

        # 1. Query-conditioned local z_set fusion. A single z_set has no
        # attention selection problem, so use a lightweight explicit fusion.
        local = z_set.unsqueeze(2).expand(-1, -1, S_q, -1)
        q_local = self.local_fusion(torch.cat([
            z_query_tokens, local, z_query_tokens * local, z_query_tokens - local,
        ], dim=-1))
        q_local = q_local.masked_fill(query_pad_mask.unsqueeze(-1), 0.0)

        # 2. Flatten queries and let them read the full temporal context.
        q_flat = q_local.view(B, T * S_q, D)
        q_flat_mask = query_pad_mask.view(B, T * S_q)
        time_2d = state_times.squeeze(-1) if state_times.dim() == 3 else state_times
        query_times = time_2d.unsqueeze(-1).expand(-1, -1, S_q).reshape(B, T * S_q)
        h = q_flat
        for layer in self.context_cross_layers:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(
                    lambda x, active_layer=layer: active_layer(
                        x, s_context, query_times, time_2d, context_pad_mask, q_flat_mask
                    ),
                    h,
                    use_reentrant=False,
                )
            else:
                h = layer(h, s_context, query_times, time_2d, context_pad_mask, q_flat_mask)

        # 3. Projection & hardening
        preds_raw = h
        preds = self.output_proj(preds_raw)
        preds = self.output_norm(preds)
        preds = preds.masked_fill(q_flat_mask.unsqueeze(-1), 0.0)
        preds = self.clamp(preds)
        return preds.view(B, T, S_q, D)

    def forward_packed(
        self,
        s_context: torch.Tensor,
        packed_queries: torch.Tensor,
        query_times: torch.Tensor,
        local_z_set: torch.Tensor,
        state_times: torch.Tensor,
        context_pad_mask: torch.Tensor,
        query_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Exact sparse-query Phase-1 predictor path.

        `packed_queries` contains only masked numeric targets, padded to the
        largest target count in the batch. Queries remain independent: each is
        fused with its own timestamp's partial z_set, then reads temporal
        context independently.
        """
        B, M, D = packed_queries.shape
        T = s_context.size(1)
        if query_times.shape != (B, M):
            raise ValueError("query_times must have shape (B, M)")
        if query_pad_mask.shape != (B, M):
            raise ValueError("query_pad_mask must have shape (B, M)")
        if M == 0:
            return packed_queries

        if local_z_set.shape != (B, M, D):
            raise ValueError("local_z_set must have shape (B, M, D)")
        q_local = self.local_fusion(torch.cat([
            packed_queries, local_z_set, packed_queries * local_z_set, packed_queries - local_z_set,
        ], dim=-1))
        q_local = q_local.masked_fill(query_pad_mask.unsqueeze(-1), 0.0)

        time_2d = state_times.squeeze(-1) if state_times.dim() == 3 else state_times
        signed_dt = time_2d.unsqueeze(1) - query_times.unsqueeze(2)
        signed_log_dt = torch.sign(signed_dt) * torch.log1p(signed_dt.abs())
        additive_key_mask = torch.zeros(
            B, 1, 1, T, dtype=packed_queries.dtype, device=packed_queries.device
        ).masked_fill(context_pad_mask.view(B, 1, 1, T), float("-inf"))
        h = q_local
        for layer in self.context_cross_layers:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(
                    lambda x, active_layer=layer: active_layer(
                        x, s_context, query_times, time_2d, context_pad_mask, query_pad_mask,
                        signed_log_dt, additive_key_mask
                    ),
                    h,
                    use_reentrant=False,
                )
            else:
                h = layer(
                    h, s_context, query_times, time_2d, context_pad_mask, query_pad_mask,
                    signed_log_dt, additive_key_mask,
                )
        preds = self.output_norm(self.output_proj(h))
        preds = preds.masked_fill(query_pad_mask.unsqueeze(-1), 0.0)
        return self.clamp(preds)


def _pairwise_soft_jaccard_probs(
    probs: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    *,
    block_size: int = 256,
) -> torch.Tensor:
    p = probs.float()
    if weights is not None:
        w = weights.float().view(1, 1, -1)
    else:
        w = None
    n = int(p.size(0))
    out: List[torch.Tensor] = []
    for start in range(0, n, int(max(1, block_size))):
        end = min(n, start + int(max(1, block_size)))
        a = p[start:end].unsqueeze(1)
        b = p.unsqueeze(0)
        mn = torch.minimum(a, b)
        mx = torch.maximum(a, b)
        if w is not None:
            mn = mn * w
            mx = mx * w
        num = mn.sum(dim=-1)
        den = mx.sum(dim=-1).clamp_min(1e-8)
        out.append(num / den)
    return torch.cat(out, dim=0)


def _assert_no_embedding_inputs(payload: Dict[str, Any]) -> None:
    assert "embedding" not in payload, "Embedding-based retrieval is forbidden"
    assert "debug_embedding" not in payload, "Embedding-based retrieval is forbidden"


def soft_jaccard_probs(
    p: torch.Tensor,
    q: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    p = p.float()
    q = q.float()
    mn = torch.minimum(p, q)
    mx = torch.maximum(p, q)
    if weights is not None:
        w = weights.float().view(-1)
        mn = mn * w
        mx = mx * w
    return mn.sum(dim=-1) / mx.sum(dim=-1).clamp_min(1e-8)


def weighted_soft_jaccard_probs(
    p: torch.Tensor,
    q: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    return soft_jaccard_probs(p, q, weights=weights)


def cosine_probs(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(p.float(), q.float(), dim=-1)


def hard_jaccard_sets(a_mask: torch.Tensor, b_mask: torch.Tensor) -> torch.Tensor:
    a = a_mask.float()
    b = b_mask.float()
    overlap = (a * b).sum(dim=-1)
    union = (a + b - a * b).sum(dim=-1).clamp_min(1.0)
    return overlap / union


# JEPA Model


class JEPAModel(pl.LightningModule):
    def __init__(
        self,
        config: RecipeConfig,
        embed_dim: int = 128,
        num_features: int = 8000,
        schema_path: str = "",
        num_heads: int = 4,
        max_lr: float = 1e-4, 
        weight_decay: float = 1e-4,
        phase1_config: Dict[str, Any] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.phase1_architecture_version = PHASE1_ARCHITECTURE_VERSION
        self.config = config
        self.embed_dim = embed_dim
        self.max_lr = max_lr
        self.wd = weight_decay
        self.phase1_config = phase1_config or {}
        self.phase1_runtime_assertions = bool(self.phase1_config.get("runtime_invariant_assertions", False))
        # Automatic Lightning accumulation divides each returned microbatch
        # loss by N.  We instead accumulate raw global-L1 sums and normalize
        # the already-DDP-reduced gradients once per optimizer cycle.
        self._phase1_cycle_global_count: Optional[torch.Tensor] = None
        self._phase1_cycle_masked_events: Optional[torch.Tensor] = None
        self.phase1_profile_ranges = False
        expected_schema_sha256 = self.phase1_config.get("schema_sha256")
        self.schema_path = _require_schema_processed_csv(schema_path, expected_schema_sha256)
        self.schema_sha256 = _schema_sha256(self.schema_path)
        self.hparams["phase1_architecture_version"] = PHASE1_ARCHITECTURE_VERSION
        self.hparams["schema_sha256"] = self.schema_sha256
        self.numeric_readout_every_n_val_batches = max(
            1, int(self.phase1_config.get("numeric_readout_every_n_val_batches", 8))
        )
        self.validation_telemetry_max_tokens = max(
            16, int(self.phase1_config.get("validation_telemetry_max_tokens", 4096))
        )
        self.validation_value_intervention_every_n_val_batches = max(
            1, int(self.phase1_config.get("validation_value_intervention_every_n_val_batches", 32))
        )
        self.validation_value_intervention_max_tokens = max(
            1, int(self.phase1_config.get("validation_value_intervention_max_tokens", 32))
        )
        self.train_telemetry_every_n_steps = max(
            1, int(self.phase1_config.get("telemetry_every_n_steps", 50))
        )
        self.register_buffer(
            "feature_type_ids",
            _load_feature_type_ids(self.schema_path, num_features, self.schema_sha256),
            persistent=True,
        )
        state_feature_ids, state_numeric_feature_ids, state_categorical_feature_ids = _load_state_feature_id_tables(
            self.schema_path, num_features, self.schema_sha256
        )
        self.register_buffer("state_feature_ids", state_feature_ids, persistent=False)
        self.register_buffer("state_numeric_feature_ids", state_numeric_feature_ids, persistent=False)
        self.register_buffer("state_categorical_feature_ids", state_categorical_feature_ids, persistent=False)
        
        # Manifold Quality Metrics
        
        # 1. Online Encoder  (receives gradients)
        self.feature_projector = FeatureProjector(
            embed_dim,
            num_features,
            self.feature_type_ids,
        )
        self.fusion_norm       = nn.LayerNorm(embed_dim)
        self.set_encoder       = SetEncoder(config, embed_dim, num_heads)

        # 2. Online Backbone
        self.state_backbone = RelativeTimeTransformerEncoder(
            embed_dim, num_heads, config.context_layers
        )
        self.context_output_norm = nn.LayerNorm(embed_dim)

        # 3. JEPA Predictor + Query Constructor
        predictor_layers    = getattr(config, 'predictor_layers', config.context_layers)
        self.z_constructor  = ZConstructor(embed_dim, self.feature_projector)
        self.predictor      = JEPAPredictor(
            embed_dim,
            num_heads,
            predictor_layers,
        )

        # 4. EMA Teacher. Full teacher family only.
        self.target_feature_projector = deepcopy(self.feature_projector)
        self.target_fusion_norm = deepcopy(self.fusion_norm)
        self.target_set_encoder = deepcopy(self.set_encoder)
        self.target_backbone = deepcopy(self.state_backbone)
        self.target_context_output_norm = deepcopy(self.context_output_norm)
        target_modules = [
            self.target_feature_projector,
            self.target_fusion_norm,
            self.target_set_encoder,
            self.target_backbone,
            self.target_context_output_norm,
        ]
        for m in target_modules:
            for p in m.parameters():
                p.requires_grad = False

        decay = config.ema_decay
        self.ema_feature_projector_updater = EMAMomentumUpdater(
            self.target_feature_projector, self.feature_projector, decay
        )
        self.ema_fusion_norm_updater = EMAMomentumUpdater(
            self.target_fusion_norm, self.fusion_norm, decay
        )
        self.ema_set_encoder_updater = EMAMomentumUpdater(
            self.target_set_encoder, self.set_encoder, decay
        )
        self.ema_backbone_updater = EMAMomentumUpdater(self.target_backbone, self.state_backbone, decay)
        self.ema_context_output_norm_updater = EMAMomentumUpdater(
            self.target_context_output_norm, self.context_output_norm, decay
        )
        object.__setattr__(self, "_compiled_state_backbone", None)
        object.__setattr__(self, "_compiled_predictor", None)

        print(f"[JEPAModel] Phase1 full EMA teacher enabled, ema_decay={decay}")

    def configure_phase1_performance(self, performance: Dict[str, Any]) -> Dict[str, Any]:
        """Apply non-semantic Phase-1 execution settings and return the effective values."""
        backend = str(performance.get("relative_attention_backend", "exact_query_chunked"))
        if backend != "exact_query_chunked":
            raise ValueError(
                "Phase 1 currently implements only the exact_query_chunked relative-time backend; "
                f"got {backend!r}."
            )
        chunk = max(1, int(performance.get("relative_query_chunk_size", 64)))
        for encoder in (self.state_backbone, self.target_backbone):
            for layer in encoder.layers:
                layer.relative_query_chunk_size = chunk
        for layer in self.predictor.context_cross_layers:
            layer.relative_query_chunk_size = chunk
        checkpointing_requested = performance.get("activation_checkpointing", "off")
        if checkpointing_requested not in {True, False, "auto", "on", "off"}:
            raise ValueError("phase1.performance.activation_checkpointing must be auto, on, off, true, or false")
        # "auto" is deliberately off until a CUDA benchmark demonstrates that
        # activation memory is the limiting resource. It never changes semantics.
        checkpointing_actual = checkpointing_requested in {True, "on"}
        self.state_backbone.activation_checkpointing = checkpointing_actual
        self.predictor.activation_checkpointing = checkpointing_actual
        compile_requested = bool(performance.get("compile_enabled", False))
        compile_actual = "disabled"
        if compile_requested and torch.cuda.is_available():
            if not hasattr(torch, "compile"):
                raise RuntimeError("phase1.performance.compile_enabled requires torch.compile support.")
            mode = str(performance.get("compile_mode", "default"))
            if mode == "reduce-overhead":
                print(
                    "[Phase 1] Ignoring compile_mode=reduce-overhead because it enables CUDA graphs; "
                    "using compile_mode=default for dynamic sparse-query shapes."
                )
                mode = "default"
            try:
                torch._inductor.config.triton.cudagraphs = False
            except (AttributeError, RuntimeError):
                # Older PyTorch versions may not expose the private Inductor
                # flag; ``default`` mode still avoids the reduce-overhead
                # CUDA-graph policy.
                pass
            try:
                object.__setattr__(self, "_compiled_state_backbone", torch.compile(self.state_backbone, mode=mode))
                # Compile the actual sparse-query method, not the module's
                # dense-reference `forward` entrypoint.
                object.__setattr__(
                    self,
                    "_compiled_predictor",
                    torch.compile(self.predictor.forward_packed, mode=mode),
                )
            except Exception as exc:
                raise RuntimeError("Phase-1 torch.compile initialization failed; refusing semantic fallback.") from exc
            compile_actual = f"regional:{mode}"
        elif compile_requested:
            compile_actual = "not_available_without_cuda"
        return {
            "relative_attention_backend_requested": backend,
            "relative_attention_backend_actual": "exact_query_chunked",
            "relative_query_chunk_size": chunk,
            "compile_requested": compile_requested,
            "compile_actual": compile_actual,
            # Phase-1 sparse-query shapes are dynamic.  CUDA graphs are
            # deliberately disabled even when regional compilation is active.
            "cuda_graphs_actual": False,
            "activation_checkpointing_requested": checkpointing_requested,
            "activation_checkpointing_actual": checkpointing_actual,
        }

    def _run_state_backbone(self, *args, **kwargs) -> torch.Tensor:
        compiled = self._compiled_state_backbone
        return (compiled or self.state_backbone)(*args, **kwargs)

    def _run_packed_predictor(self, *args, **kwargs) -> torch.Tensor:
        compiled = self._compiled_predictor
        return (compiled or self.predictor.forward_packed)(*args, **kwargs)

    def on_train_epoch_start(self) -> None:
        """Lightning-owned epoch hook for deterministic length-bucket reshuffling."""
        self._phase1_cycle_global_count = None
        self._phase1_cycle_masked_events = None
        datamodule = getattr(self.trainer, "datamodule", None)
        sampler = getattr(datamodule, "_train_batch_sampler", None)
        if sampler is not None:
            sampler.set_epoch(int(self.current_epoch))

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        hparams = checkpoint.get("hyper_parameters", {})
        saved = hparams.get("phase1_architecture_version")
        if saved != PHASE1_ARCHITECTURE_VERSION:
            raise RuntimeError(
                "Phase1 checkpoint architecture mismatch: expected "
                f"{PHASE1_ARCHITECTURE_VERSION!r}, found {saved!r}. "
                "Phase 1 checkpoints with a different target/predictor graph must be trained from scratch."
            )
        saved_schema_sha256 = hparams.get("schema_sha256")
        if not saved_schema_sha256:
            raise RuntimeError("Phase1 checkpoint is missing required schema_sha256 hyperparameter.")
        if str(saved_schema_sha256) != self.schema_sha256:
            raise RuntimeError(
                "Phase1 checkpoint schema hash mismatch: "
                f"checkpoint={saved_schema_sha256}, current={self.schema_sha256}."
            )
        binding = checkpoint.get("phase1_schema_binding")
        if not isinstance(binding, dict):
            raise RuntimeError("Phase1 checkpoint is missing required schema binding metadata.")
        required = {"schema_sha256", "num_features", "feature_type_ids_shape", "feature_type_ids_sha256"}
        missing = sorted(required - set(binding))
        if missing:
            raise RuntimeError(f"Phase1 checkpoint schema binding is incomplete: missing {missing}.")
        if str(binding["schema_sha256"]) != self.schema_sha256:
            raise RuntimeError(
                "Phase1 checkpoint schema hash mismatch: "
                f"checkpoint={binding['schema_sha256']}, current={self.schema_sha256}."
            )
        if int(binding["num_features"]) != int(self.feature_type_ids.numel() - 1):
            raise RuntimeError("Phase1 checkpoint num_features does not match current schema/model vocabulary.")
        if list(binding["feature_type_ids_shape"]) != list(self.feature_type_ids.shape):
            raise RuntimeError("Phase1 checkpoint feature type table shape mismatch.")
        if str(binding["feature_type_ids_sha256"]) != _tensor_sha256(self.feature_type_ids):
            raise RuntimeError("Phase1 checkpoint feature type table content mismatch.")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["phase1_schema_binding"] = {
            "schema_sha256": self.schema_sha256,
            "num_features": int(self.feature_type_ids.numel() - 1),
            "feature_type_ids_shape": list(self.feature_type_ids.shape),
            "feature_type_ids_sha256": _tensor_sha256(self.feature_type_ids),
        }

    def train(self, mode: bool = True):
        super().train(mode)
        for module in (
            self.target_feature_projector,
            self.target_fusion_norm,
            self.target_set_encoder,
            self.target_backbone,
            self.target_context_output_norm,
        ):
            module.eval()
        return self



    def _encode_event_tokens(
        self,
        code: torch.Tensor,
        val: torch.Tensor,
        projector: Optional[FeatureProjector] = None,
        fusion_norm: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Time-free Phase-1 fact encoding followed by post-projection norm."""
        fp = projector or self.feature_projector
        norm = fusion_norm or self.fusion_norm
        return norm(fp(code, val))

    def _teacher_frontend_modules(self):
        """Return the time-free EMA teacher front-end used by Phase 1."""
        return (
            self.target_feature_projector,
            self.target_fusion_norm,
            self.target_set_encoder,
        )

    def _encode_valid_timestamp_sets(
        self,
        state_tokens: torch.Tensor,
        state_values: torch.Tensor,
        token_pad_mask: torch.Tensor,
        seq_pad: torch.Tensor,
        *,
        projector: FeatureProjector,
        fusion_norm: nn.Module,
        set_encoder: SetEncoder,
    ) -> Dict[str, torch.Tensor]:
        """Run the event/set frontend only for non-padding timestamps.

        The returned `z_set` is dense because temporal attention requires a
        timestamp grid; token states remain packed by valid timestamp and are
        only gathered for actual masked queries.
        """
        B, T, S = state_tokens.shape
        D = self.embed_dim
        valid_bt = (~seq_pad).nonzero(as_tuple=False)
        timestamp_to_valid = torch.full((B, T), -1, device=state_tokens.device, dtype=torch.long)
        if valid_bt.numel() == 0:
            # No SetEncoder invocation exists from which to infer an autocast
            # dtype. This tensor is fully padded and cannot affect a real loss.
            z_set = next(set_encoder.parameters()).new_zeros((B, T, D), device=state_tokens.device)
            return {
                "z_set": z_set,
                "set_tokens": z_set.new_zeros((0, S, D)),
                "event_tokens": z_set.new_zeros((0, S, D)),
                "valid_bt": valid_bt,
                "timestamp_to_valid": timestamp_to_valid,
                "token_pad_mask": token_pad_mask.new_ones((0, S)),
            }
        b_idx, t_idx = valid_bt.unbind(dim=1)
        valid_tokens = state_tokens[b_idx, t_idx]
        valid_values = state_values[b_idx, t_idx]
        valid_pad = token_pad_mask[b_idx, t_idx]
        event_tokens = self._encode_event_tokens(
            valid_tokens, valid_values, projector=projector, fusion_norm=fusion_norm
        ).masked_fill(valid_pad.unsqueeze(-1), 0.0)
        pooled, set_tokens = set_encoder(event_tokens, valid_pad, return_tokens=True)
        # Under autocast SetEncoder output may differ from input/value dtype.
        # Allocate every scatter destination from the actual source tensor.
        z_set = pooled.new_zeros((B, T, D))
        z_set[b_idx, t_idx] = pooled.squeeze(1)
        timestamp_to_valid[b_idx, t_idx] = torch.arange(valid_bt.size(0), device=state_tokens.device)
        return {
            "z_set": z_set,
            "set_tokens": set_tokens,
            "event_tokens": event_tokens,
            "valid_bt": valid_bt,
            "timestamp_to_valid": timestamp_to_valid,
            "token_pad_mask": valid_pad,
        }

    @staticmethod
    def _distributed_sum(value: torch.Tensor) -> torch.Tensor:
        total = value.detach().clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        return total

    def _global_masked_l1(
        self,
        pred_valid: torch.Tensor,
        target_valid: torch.Tensor,
        *,
        with_grad: bool,
        anchor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Global masked-element L1 mean with DDP-correct gradient scaling."""
        local_sum = F.l1_loss(pred_valid, target_valid, reduction="sum") if pred_valid.numel() else anchor * 0.0
        local_count = torch.tensor(float(pred_valid.numel()), device=anchor.device, dtype=torch.float32)
        global_sum = self._distributed_sum(local_sum)
        global_count = self._distributed_sum(local_count)
        if with_grad:
            world_size = (
                torch.distributed.get_world_size()
                if torch.distributed.is_available() and torch.distributed.is_initialized() else 1
            )
            # DDP averages rank gradients. Scaling each local sum by world_size
            # recovers grad(global_sum / global_element_count).
            loss = (local_sum + anchor * 0.0) * (float(world_size) / global_count.clamp_min(1.0))
        else:
            loss = global_sum / global_count.clamp_min(1.0)
        return loss, global_sum, global_count

    def _ddp_zero_target_anchor(self) -> torch.Tensor:
        """Touch every online parameter when a rank has no packed targets."""
        anchor = self.context_output_norm.weight.reshape(-1)[0] * 0.0
        for parameter in self.parameters():
            if parameter.requires_grad:
                anchor = anchor + parameter.reshape(-1)[0] * 0.0
        return anchor

    def _gather_global_timestamp_sample(self, local: torch.Tensor) -> torch.Tensor:
        """Bounded all-rank sample for global effective-rank telemetry."""
        limit = max(1, int(self.validation_telemetry_max_tokens))
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return local[:limit]
        world_size = torch.distributed.get_world_size()
        per_rank = max(1, int(math.ceil(limit / world_size)))
        local = local[:per_rank]
        count = torch.tensor([local.size(0)], device=local.device, dtype=torch.long)
        counts = [torch.zeros_like(count) for _ in range(world_size)]
        torch.distributed.all_gather(counts, count)
        max_rows = max(int(item.item()) for item in counts)
        padded = local.new_zeros((max_rows, local.size(-1)))
        if local.numel():
            padded[:local.size(0)] = local
        gathered = [torch.zeros_like(padded) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, padded)
        rows = [tensor[:int(n.item())] for tensor, n in zip(gathered, counts)]
        return torch.cat(rows, dim=0)[:limit] if rows else local

    @staticmethod
    def _scatter_valid_timestamp_tokens(
        frontend: Dict[str, torch.Tensor], B: int, T: int, key: str = "set_tokens"
    ) -> torch.Tensor:
        tokens = frontend[key]
        S, D = tokens.size(1), tokens.size(2)
        dense = tokens.new_zeros((B, T, S, D))
        valid_bt = frontend["valid_bt"]
        if valid_bt.numel():
            dense[valid_bt[:, 0], valid_bt[:, 1]] = tokens
        return dense

    def encode_teacher_sets(
        self,
        state_tokens: torch.Tensor,
        state_values: torch.Tensor,
        state_pad_mask: torch.Tensor,
        seq_pad_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """EMA SetEncoder states for downstream and internal diagnostics only."""
        fp, fnorm, senc = self._teacher_frontend_modules()
        return self._encode_valid_timestamp_sets(
            state_tokens, state_values, state_pad_mask, seq_pad_mask,
            projector=fp, fusion_norm=fnorm, set_encoder=senc,
        )

    def contextualize_teacher_z_set(
        self,
        z_set: torch.Tensor,
        seq_pad: torch.Tensor,
        state_times: torch.Tensor,
        *,
        causal_mask: Optional[Any] = None,
    ) -> torch.Tensor:
        """Apply the EMA relative-time backbone and required outer z_set skip."""
        context_delta = self.target_backbone(
            z_set, seq_pad, state_times, causal_mask=causal_mask
        )
        z_context = self.target_context_output_norm(z_set + context_delta)
        return z_context.masked_fill(seq_pad.unsqueeze(-1), 0.0)

    def _encode_teacher_states(
        self,
        state_tokens: torch.Tensor,
        state_values: torch.Tensor,
        s_pad: torch.Tensor,
        seq_pad: torch.Tensor,
        time: torch.Tensor,
        causal_mask: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode internal post-SetEncoder teacher member states, time-free visit
        embeddings, and EMA relative-time contextual states. Member states are
        internal SetEncoder diagnostics only; the Phase-1 JEPA target is the
        EMA pre-SetEncoder z_event emitted by `_encode_teacher_z_event`.
        """
        B, T, _ = state_tokens.shape
        frontend = self.encode_teacher_sets(
            state_tokens, state_values, s_pad, seq_pad
        )
        z_visit_teacher = frontend["z_set"]
        teacher_set_member_states = self._scatter_valid_timestamp_tokens(frontend, B, T)
        teacher_z_context = self.contextualize_teacher_z_set(
            z_visit_teacher, seq_pad, time, causal_mask=causal_mask
        )
        return teacher_set_member_states, z_visit_teacher, teacher_z_context

    def _pack_masked_queries(
        self,
        state_tokens: torch.Tensor,
        is_masked: torch.Tensor,
        state_times: torch.Tensor,
        prepacked: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Pack exactly the true masked numeric targets as B×M rather than B×T×S."""
        B = state_tokens.size(0)
        D = self.embed_dim
        if prepacked is not None:
            qpad = prepacked["packed_query_pad_mask"].bool()
            vids = prepacked["packed_query_vids"].long()
            z_id = self.feature_projector.encode_identity(vids)
            query_values = z_id + self.z_constructor.query_phi(z_id)
            query_values = query_values.masked_fill(qpad.unsqueeze(-1), 0.0)
            return {
                "queries": query_values,
                "query_pad_mask": qpad,
                "timestamp_indices": prepacked["packed_query_timestamp_indices"].long(),
                "slot_indices": prepacked["packed_query_slot_indices"].long(),
                "query_times": prepacked["packed_query_times"],
                "indices": prepacked["packed_query_indices"].long(),
                "ranks": prepacked["packed_query_ranks"].long(),
            }
        indices = is_masked.nonzero(as_tuple=False)
        counts = torch.bincount(indices[:, 0], minlength=B) if indices.numel() else torch.zeros(B, device=state_tokens.device, dtype=torch.long)
        M = int(counts.max().detach()) if counts.numel() else 0
        qpad = torch.ones((B, M), dtype=torch.bool, device=state_tokens.device)
        packed_t = torch.zeros((B, M), dtype=torch.long, device=state_tokens.device)
        packed_s = torch.zeros((B, M), dtype=torch.long, device=state_tokens.device)
        if M == 0:
            return {
                "queries": state_tokens.new_zeros((B, 0, D), dtype=torch.float32),
                "query_pad_mask": qpad,
                "timestamp_indices": packed_t,
                "slot_indices": packed_s,
                "query_times": state_tokens.new_zeros((B, 0), dtype=torch.float32),
                "indices": indices,
                "ranks": torch.zeros((0,), dtype=torch.long, device=state_tokens.device),
            }
        rank = torch.arange(indices.size(0), device=state_tokens.device) - torch.repeat_interleave(
            torch.cumsum(counts, dim=0) - counts, counts
        )
        b_idx, t_idx, s_idx = indices.unbind(dim=1)
        vids = state_tokens[b_idx, t_idx, s_idx]
        z_id = self.feature_projector.encode_identity(vids)
        query_values = z_id + self.z_constructor.query_phi(z_id)
        packed_query = query_values.new_zeros((B, M, D))
        packed_query[b_idx, rank] = query_values
        qpad[b_idx, rank] = False
        packed_t[b_idx, rank] = t_idx
        packed_s[b_idx, rank] = s_idx
        time_2d = state_times.squeeze(-1) if state_times.dim() == 3 else state_times
        query_times = time_2d.new_zeros((B, M))
        query_times[b_idx, rank] = time_2d[b_idx, t_idx]
        return {
            "queries": packed_query,
            "query_pad_mask": qpad,
            "timestamp_indices": packed_t,
            "slot_indices": packed_s,
            "query_times": query_times,
            "indices": indices,
            "ranks": rank,
        }

    @staticmethod
    def _gather_packed_z_set(
        z_set: torch.Tensor,
        query_timestamps: torch.Tensor,
        query_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, M = query_timestamps.shape
        if M == 0:
            return z_set.new_zeros((B, 0, z_set.size(-1)))
        batch_idx = torch.arange(B, device=query_timestamps.device).unsqueeze(1).expand(B, M)
        return z_set[batch_idx, query_timestamps].masked_fill(query_pad_mask.unsqueeze(-1), 0.0)

    def _assert_prepacked_query_consistency(
        self,
        batch: Dict[str, torch.Tensor],
        is_masked: torch.Tensor,
        packed: Dict[str, torch.Tensor],
        state_tokens: torch.Tensor,
        state_values: torch.Tensor,
        state_times: torch.Tensor,
    ) -> None:
        """Debug-only validation of collate-produced packed numeric queries."""
        required = ("packed_query_true_values",)
        missing = [key for key in required if key not in batch]
        if missing:
            raise AssertionError(f"packed query batch is missing required fields: {missing}")
        numeric_masked = is_masked & (self.feature_type_ids[state_tokens] == FEATURE_TYPE_NUMERIC)
        expected_indices = numeric_masked.nonzero(as_tuple=False)
        indices = packed["indices"]
        ranks = packed["ranks"]
        qpad = packed["query_pad_mask"]
        if indices.ndim != 2 or indices.size(-1) != 3:
            raise AssertionError("packed_query_indices must have shape (Q, 3)")
        if not torch.equal(indices, expected_indices):
            raise AssertionError("packed query indices do not match numeric state_mask_labels")
        if ranks.shape != (indices.size(0),):
            raise AssertionError("packed_query_ranks must have shape (Q,)")
        if ranks.numel() and not bool(((ranks >= 0) & (ranks < qpad.size(1))).all()):
            raise AssertionError("packed_query_ranks contains an out-of-range packed position")
        if not torch.equal((~qpad).sum(), numeric_masked.sum()):
            raise AssertionError("packed valid query count does not match numeric masked count")
        if not indices.numel():
            return
        b_idx, t_idx, s_idx = indices.unbind(dim=1)
        if not torch.equal(packed["timestamp_indices"][b_idx, ranks], t_idx):
            raise AssertionError("packed query timestamp indices do not match state_mask_labels")
        if not torch.equal(packed["slot_indices"][b_idx, ranks], s_idx):
            raise AssertionError("packed query slot indices do not match state_mask_labels")
        if qpad[b_idx, ranks].any():
            raise AssertionError("packed valid query index points to padding")
        if not torch.equal(batch["packed_query_vids"][b_idx, ranks], state_tokens[b_idx, t_idx, s_idx]):
            raise AssertionError("packed query VIDs do not match state_tokens")
        if not torch.equal(batch["packed_query_true_values"][b_idx, ranks], state_values[b_idx, t_idx, s_idx, 0]):
            raise AssertionError("packed query values do not match state_values")
        time_2d = state_times.squeeze(-1) if state_times.dim() == 3 else state_times
        if not torch.equal(packed["query_times"][b_idx, ranks], time_2d[b_idx, t_idx]):
            raise AssertionError("packed query times do not match state_times")

    def _encode_teacher_z_event(
        self,
        packed_vids: torch.Tensor,
        packed_values: torch.Tensor,
        query_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """EMA pre-SetEncoder, time-free z_event target for packed queries."""
        with torch.no_grad():
            z_event = self._encode_event_tokens(
                packed_vids, packed_values.unsqueeze(-1),
                projector=self.target_feature_projector, fusion_norm=self.target_fusion_norm,
            )
        return z_event.masked_fill(query_pad_mask.unsqueeze(-1), 0.0)

    def _forward_phase1_numeric_jepa(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        output_mode: str = "loss_only",
    ) -> Dict[str, torch.Tensor]:
        """Authoritative sparse-query Phase-1 graph shared by train and validation."""
        if output_mode not in {"loss_only", "readout_cache", "telemetry", "full_debug"}:
            raise ValueError("output_mode must be loss_only, readout_cache, telemetry, or full_debug")
        state_tokens = batch["state_tokens"]
        state_values = batch["state_values"]
        s_pad = batch["state_pad_mask"].bool()
        seq_pad = batch["seq_pad_mask"].bool()
        state_times = batch["state_times"]
        mask_labels = batch["state_mask_labels"]
        B, T, S_s = state_tokens.shape

        is_masked = (mask_labels > 0.5).squeeze(-1) & (~s_pad) & (~seq_pad.unsqueeze(-1))
        if self.phase1_runtime_assertions:
            masked_types = self.feature_type_ids[state_tokens[is_masked]]
            if masked_types.numel() and not torch.all(masked_types == FEATURE_TYPE_NUMERIC):
                raise AssertionError("Phase 1 only permits masked numeric state events")

        # The masked event is removed before FeatureProjector: neither VID nor
        # value can enter the student context path.
        student_pad = s_pad | is_masked | seq_pad.unsqueeze(-1)
        student_tokens = state_tokens.masked_fill(student_pad, 0)
        student_values = state_values.masked_fill(student_pad.unsqueeze(-1), 0.0)
        range_ctx = torch.profiler.record_function if self.phase1_profile_ranges else contextlib.nullcontext
        with range_ctx("phase1/student_frontend"):
            student_frontend = self._encode_valid_timestamp_sets(
                student_tokens, student_values, student_pad, seq_pad,
                projector=self.feature_projector, fusion_norm=self.fusion_norm, set_encoder=self.set_encoder,
            )
        student_z_set = student_frontend["z_set"]
        with range_ctx("phase1/temporal_backbone"):
            context_delta = self._run_state_backbone(student_z_set, seq_pad, state_times, causal_mask=None)
            student_z_context = self.context_output_norm(student_z_set + context_delta)
            student_z_context = student_z_context.masked_fill(seq_pad.unsqueeze(-1), 0.0)

        prepacked_keys = {
            "packed_query_vids", "packed_query_timestamp_indices", "packed_query_slot_indices",
            "packed_query_times", "packed_query_pad_mask", "packed_query_indices", "packed_query_ranks",
        }
        prepacked = batch if prepacked_keys.issubset(batch) else None
        packed = self._pack_masked_queries(state_tokens, is_masked, state_times, prepacked=prepacked)
        if self.phase1_runtime_assertions and prepacked is not None:
            self._assert_prepacked_query_consistency(
                batch, is_masked, packed, state_tokens, state_values, state_times
            )
        local_z_set = self._gather_packed_z_set(
            student_z_set, packed["timestamp_indices"], packed["query_pad_mask"]
        )
        with range_ctx("phase1/packed_predictor"):
            z_pred_packed = self._run_packed_predictor(
                student_z_context,
                packed["queries"],
                packed["query_times"],
                local_z_set,
                state_times,
                seq_pad,
                packed["query_pad_mask"],
            )

        packed_true_values = (
            batch["packed_query_true_values"].to(dtype=state_values.dtype)
            if prepacked is not None else state_values.new_zeros(packed["query_pad_mask"].shape)
        )
        indices = packed["indices"]
        if prepacked is None and indices.numel():
            b_idx, t_idx, s_idx = indices.unbind(dim=1)
            packed_true_values[b_idx, packed["ranks"]] = state_values[b_idx, t_idx, s_idx, 0]
        target_vids = batch["packed_query_vids"].long() if prepacked is not None else state_tokens.new_zeros(packed["query_pad_mask"].shape)
        if prepacked is None and indices.numel():
            target_vids[b_idx, packed["ranks"]] = state_tokens[b_idx, t_idx, s_idx]
        with range_ctx("phase1/teacher_event_target"):
            target_packed = self._encode_teacher_z_event(
                target_vids,
                packed_true_values,
                packed["query_pad_mask"],
            )
        valid_mask = ~packed["query_pad_mask"]
        pred_valid = z_pred_packed[valid_mask]
        target_valid = target_packed.detach()[valid_mask]
        out = {
            "loss_inputs": {
                "pred_valid": pred_valid,
                "target_valid": target_valid,
                "valid_mask": valid_mask,
            },
        }
        if output_mode != "loss_only":
            packed_valid = ~packed["query_pad_mask"]
            local_z_set_norm_sum = local_z_set.norm(dim=-1).masked_select(packed_valid).sum() if packed_valid.any() else student_z_context.sum() * 0.0
            local_z_set_norm_count = packed_valid.sum()
            out.update({
            # ``indices`` is ordered identically to packed valid predictions
            # (batch-major, then timestamp/slot order). Validation uses this
            # to de-duplicate DistributedSampler padding by admission ID.
            "packed_batch_indices": packed["indices"][:, 0],
            "packed_indices": packed["indices"],
            "z_pred_packed": z_pred_packed,
            "packed_query_pad_mask": packed["query_pad_mask"],
            "packed_timestamp_indices": packed["timestamp_indices"],
            "packed_slot_indices": packed["slot_indices"],
            "event_valid_mask": is_masked,
            "student_z_set": student_z_set,
            "student_z_context": student_z_context,
            "local_z_set_packed": local_z_set,
            "local_z_set_norm_sum": local_z_set_norm_sum,
            "local_z_set_norm_count": local_z_set_norm_count,
            "target_z_event_packed": target_packed,
            })
            out["packed_true_values"] = packed_true_values
        if output_mode in {"telemetry", "full_debug"}:
            # Teacher SetEncoder output is downstream/diagnostic-only. Dense
            # B×T×S×D scatter remains reserved for explicit full_debug calls.
            with torch.no_grad():
                teacher_frontend = self.encode_teacher_sets(state_tokens, state_values, s_pad, seq_pad)
            out["teacher_z_set"] = teacher_frontend["z_set"]
        if output_mode == "full_debug":
            out["student_set_tokens"] = self._scatter_valid_timestamp_tokens(student_frontend, B, T)
            out["student_event_tokens"] = self._scatter_valid_timestamp_tokens(
                student_frontend, B, T, key="event_tokens"
            )
            out["teacher_set_member_tokens"] = self._scatter_valid_timestamp_tokens(teacher_frontend, B, T)
            out["teacher_preset_event_tokens"] = self._encode_event_tokens(
                state_tokens, state_values, projector=self.target_feature_projector, fusion_norm=self.target_fusion_norm,
            ).masked_fill(s_pad.unsqueeze(-1), 0.0).detach()
            dense_pred = z_pred_packed.new_zeros((B, T, S_s, self.embed_dim))
            indices = packed["indices"]
            if indices.numel():
                b_idx, t_idx, s_idx = indices.unbind(dim=1)
                dense_pred[b_idx, t_idx, s_idx] = z_pred_packed[b_idx, packed["ranks"]]
            out["z_pred_tokens"] = dense_pred
            out["query_pad_mask"] = ~is_masked
            out["teacher_z_context"] = self.contextualize_teacher_z_set(
                teacher_frontend["z_set"], seq_pad, state_times
            )
        return out

    def _forward_phase1_numeric_jepa_dense_reference(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Dense-query reference retained exclusively for FP32 parity tests.

        It is intentionally not called by training or validation. It preserves
        the prior B×T×S query layout so masked positions can be compared to the
        sparse production path exactly.
        """
        state_tokens = batch["state_tokens"]
        state_values = batch["state_values"]
        s_pad = batch["state_pad_mask"].bool()
        seq_pad = batch["seq_pad_mask"].bool()
        state_times = batch["state_times"]
        B, T, _ = state_tokens.shape
        is_masked = (batch["state_mask_labels"] > 0.5).squeeze(-1) & (~s_pad) & (~seq_pad.unsqueeze(-1))
        student_pad = s_pad | is_masked | seq_pad.unsqueeze(-1)
        student_tokens = state_tokens.masked_fill(student_pad, 0)
        student_values = state_values.masked_fill(student_pad.unsqueeze(-1), 0.0)
        frontend = self._encode_valid_timestamp_sets(
            student_tokens, student_values, student_pad, seq_pad,
            projector=self.feature_projector, fusion_norm=self.fusion_norm, set_encoder=self.set_encoder,
        )
        z_set = frontend["z_set"]
        delta = self._run_state_backbone(z_set, seq_pad, state_times, causal_mask=None)
        z_context = self.context_output_norm(z_set + delta).masked_fill(seq_pad.unsqueeze(-1), 0.0)
        query_tokens, query_pad_mask = self.z_constructor(state_tokens, is_masked, s_pad | seq_pad.unsqueeze(-1))
        z_pred_tokens = self.predictor(
            z_context, query_tokens, z_set, state_times, seq_pad, query_pad_mask
        )
        with torch.no_grad():
            teacher_tokens = self._encode_event_tokens(
                state_tokens, state_values, projector=self.target_feature_projector, fusion_norm=self.target_fusion_norm,
            ).masked_fill(s_pad.unsqueeze(-1), 0.0)
        return {
            "z_pred_tokens": z_pred_tokens,
            "teacher_preset_event_tokens": teacher_tokens,
            "student_z_set": z_set,
            "student_z_context": z_context,
            "valid_mask": is_masked,
            "loss": F.l1_loss(z_pred_tokens[is_masked], teacher_tokens[is_masked]) if is_masked.any() else z_context.sum() * 0.0,
        }

    @torch.no_grad()
    def predict_masked_numeric_latents(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Public frozen inference entrypoint for imputation and diagnostics."""
        return self._forward_phase1_numeric_jepa(batch, output_mode="full_debug")

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        batch_size = len(batch["hadm_id"])
        if "masked_targets_per_patient" in batch:
            local_masked_events = batch["masked_targets_per_patient"].sum(dtype=torch.float32)
        else:  # Direct unit-test/debug batches bypass the production collate path.
            local_masked_events = (
                (batch["state_mask_labels"] > 0.5).squeeze(-1)
                & (~batch["state_pad_mask"].bool())
                & (~batch["seq_pad_mask"].bool().unsqueeze(-1))
            ).sum(dtype=torch.float32)
        potential_count = self._distributed_sum(local_masked_events * float(self.embed_dim))
        sequence_lengths = (~batch["seq_pad_mask"].bool()).sum(dim=1)
        emit_telemetry = int(getattr(self, "global_step", 0)) % self.train_telemetry_every_n_steps == 0
        if emit_telemetry:
            self.log("train/masked_numeric_events", local_masked_events, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log("train/sequence_timestamps_max", sequence_lengths.max(), on_step=True, on_epoch=True, batch_size=batch_size)
            self.log("train/sequence_timestamps_mean", sequence_lengths.float().mean(), on_step=True, on_epoch=True, batch_size=batch_size)
        # A zero-target batch has identically zero loss and gradients. Skip the
        # expensive student/teacher/predictor graph only when every DDP rank
        # agrees that there are no numeric targets; otherwise each rank must
        # execute the same graph for DDP correctness.
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if not distributed and int(batch.get("max_masked_targets_in_batch", 0)) == 0:
            anchor = self._ddp_zero_target_anchor()
            zero_count = potential_count.detach()
            self._phase1_cycle_global_count = (
                zero_count if self._phase1_cycle_global_count is None
                else self._phase1_cycle_global_count + zero_count
            )
            zero_events = zero_count / float(self.embed_dim)
            self._phase1_cycle_masked_events = (
                zero_events if self._phase1_cycle_masked_events is None
                else self._phase1_cycle_masked_events + zero_events
            )
            self.log("train/loss", anchor, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False, batch_size=batch_size)
            if emit_telemetry:
                self.log("train/z_pred_norm", anchor, on_step=True, on_epoch=True, batch_size=batch_size)
                self.log("train/num_masked_numeric_events", zero_events, on_step=True, on_epoch=True, sync_dist=False, batch_size=batch_size)
                self.log("train/num_masked_latent_elements", zero_count, on_step=True, on_epoch=True, sync_dist=False, batch_size=batch_size)
                self.log("train/zero_target_batch", torch.ones((), device=self.device), on_step=True, on_epoch=True, batch_size=batch_size)
            return anchor
        out = self._forward_phase1_numeric_jepa(batch, output_mode="loss_only")
        pred_valid = out["loss_inputs"]["pred_valid"]
        target_valid = out["loss_inputs"]["target_valid"]
        anchor = pred_valid.sum() * 0.0 if pred_valid.numel() else self._ddp_zero_target_anchor()
        loss, global_loss_sum, global_count = self._global_masked_l1(
            pred_valid, target_valid, with_grad=True, anchor=anchor
        )
        count_for_cycle = global_count.detach()
        self._phase1_cycle_global_count = (
            count_for_cycle if self._phase1_cycle_global_count is None
            else self._phase1_cycle_global_count + count_for_cycle
        )
        masked_events = global_count / float(self.embed_dim)
        self._phase1_cycle_masked_events = (
            masked_events if self._phase1_cycle_masked_events is None
            else self._phase1_cycle_masked_events + masked_events
        )
        local_norm_sum = pred_valid.norm(dim=-1).sum() if pred_valid.numel() else anchor * 0.0
        global_norm_sum = self._distributed_sum(local_norm_sum)
        pred_norm = global_norm_sum / masked_events.clamp_min(1.0)
        self.log("train/loss", global_loss_sum / global_count.clamp_min(1.0), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False, batch_size=batch_size)
        if emit_telemetry:
            self.log("train/z_pred_norm", pred_norm, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log("train/num_masked_numeric_events", masked_events, on_step=True, on_epoch=True, sync_dist=False, batch_size=batch_size)
            self.log("train/num_masked_latent_elements", global_count, on_step=True, on_epoch=True, sync_dist=False, batch_size=batch_size)
            self.log("train/zero_target_batch", torch.zeros((), device=self.device), on_step=True, on_epoch=True, batch_size=batch_size)
        trainer = getattr(self, "_trainer", None)
        accumulation = int(getattr(trainer, "accumulate_grad_batches", self.phase1_config.get("accumulate_grad_batches", 1)))
        # Lightning divides the returned loss by ``accumulation`` before each
        # backward call.  Multiplying by global_count reconstructs the local
        # L1 sum; the extra factor restores DDP's averaged gradients to the
        # global sum.  on_before_optimizer_step divides once by the complete
        # cycle count, exactly matching a single global masked-element mean.
        return loss * global_count * float(max(1, accumulation))

    def on_validation_epoch_start(self) -> None:
        device = self.device
        self._val_loss_sum = torch.zeros((), device=device)
        self._val_masked_count = torch.zeros((), device=device)
        self._val_cosine_sum = torch.zeros((), device=device)
        self._val_local_norm_sum = torch.zeros((), device=device)
        self._val_local_norm_count = torch.zeros((), device=device)
        self._val_context_samples: List[torch.Tensor] = []
        self._val_zset_samples: List[torch.Tensor] = []
        self._val_sample_ids: List[str] = []
        self._val_sample_loss_sums: List[torch.Tensor] = []
        self._val_sample_counts: List[torch.Tensor] = []
        self._val_sample_cosine_sums: List[torch.Tensor] = []
        self._val_value_intervention_records: List[Dict[str, float]] = []

    @torch.no_grad()
    def _collect_validation_value_intervention(
        self, batch: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor]
    ) -> None:
        """Bounded actual-vs-zero retention audit on deterministic val targets."""
        remaining = self.validation_value_intervention_max_tokens - len(self._val_value_intervention_records)
        if remaining <= 0:
            return
        indices = out["packed_indices"]
        if not indices.numel():
            return
        chosen=[]; seen={str(row["sample_id"]) for row in self._val_value_intervention_records}
        for index in indices.tolist():
            sample_id=str(batch["hadm_id"][index[0]])
            if sample_id not in seen:
                chosen.append(index);seen.add(sample_id)
            if len(chosen) >= remaining:
                break
        if not chosen:
            return
        chosen_t=torch.tensor(chosen,device=self.device,dtype=torch.long)
        b_idx,t_idx,s_idx=chosen_t.unbind(dim=1)
        tokens=batch["state_tokens"]
        values=batch["state_values"]
        s_pad=batch["state_pad_mask"].bool()
        seq_pad=batch["seq_pad_mask"].bool()
        zero_values=values.clone()
        zero_values[b_idx,t_idx,s_idx,0]=0.0
        teacher_actual=self.encode_teacher_sets(tokens,values,s_pad,seq_pad)
        teacher_zero=self.encode_teacher_sets(tokens,zero_values,s_pad,seq_pad)
        online_actual=self._encode_valid_timestamp_sets(
            tokens,values,s_pad,seq_pad,projector=self.feature_projector,
            fusion_norm=self.fusion_norm,set_encoder=self.set_encoder,
        )
        online_zero=self._encode_valid_timestamp_sets(
            tokens,zero_values,s_pad,seq_pad,projector=self.feature_projector,
            fusion_norm=self.fusion_norm,set_encoder=self.set_encoder,
        )

        def token_at(frontend: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
            valid_index=frontend["timestamp_to_valid"][b_idx,t_idx]
            return frontend[key][valid_index,s_idx]

        teacher_preset_actual=self._encode_event_tokens(
            tokens[b_idx,t_idx,s_idx],values[b_idx,t_idx,s_idx],
            projector=self.target_feature_projector,fusion_norm=self.target_fusion_norm,
        )
        teacher_preset_zero=self._encode_event_tokens(
            tokens[b_idx,t_idx,s_idx],zero_values[b_idx,t_idx,s_idx],
            projector=self.target_feature_projector,fusion_norm=self.target_fusion_norm,
        )
        online_projector_actual=self.feature_projector(tokens[b_idx,t_idx,s_idx],values[b_idx,t_idx,s_idx])
        online_projector_zero=self.feature_projector(tokens[b_idx,t_idx,s_idx],zero_values[b_idx,t_idx,s_idx])
        deltas={
            "teacher_preset_event_delta_norm":(teacher_preset_actual-teacher_preset_zero).float().norm(dim=-1),
            "teacher_set_member_delta_norm":(token_at(teacher_actual,"set_tokens")-token_at(teacher_zero,"set_tokens")).float().norm(dim=-1),
            "online_projector_delta_norm":(online_projector_actual-online_projector_zero).float().norm(dim=-1),
            "online_set_member_delta_norm":(token_at(online_actual,"set_tokens")-token_at(online_zero,"set_tokens")).float().norm(dim=-1),
        }
        for row_index, sample_index in enumerate(b_idx.tolist()):
            record={"sample_id":str(batch["hadm_id"][sample_index])}
            for name,value in deltas.items():
                record[name]=float(value[row_index].detach().cpu())
            record["teacher_set_member_over_preset_event"]=(
                record["teacher_set_member_delta_norm"]/(record["teacher_preset_event_delta_norm"]+1e-12)
            )
            record["online_set_member_over_projector"]=(
                record["online_set_member_delta_norm"]/(record["online_projector_delta_norm"]+1e-12)
            )
            self._val_value_intervention_records.append(record)

    def _gather_value_intervention_records(self) -> List[Dict[str, float]]:
        payloads=[self._val_value_intervention_records]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            payloads=[None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(payloads,self._val_value_intervention_records)
        unique={}
        for payload in payloads:
            for row in payload:
                sample_id=str(row["sample_id"])
                if sample_id in unique and unique[sample_id] != row:
                    raise RuntimeError(f"Duplicate validation value-intervention sample disagrees: {sample_id}")
                unique[sample_id]=row
        return list(unique.values())[:self.validation_value_intervention_max_tokens]

    @torch.no_grad()
    def _teacher_online_parameter_distance(self) -> Dict[str, torch.Tensor]:
        pairs={
            "feature_projector":(self.feature_projector,self.target_feature_projector),
            "fusion_norm":(self.fusion_norm,self.target_fusion_norm),
            "set_encoder":(self.set_encoder,self.target_set_encoder),
            "backbone":(self.state_backbone,self.target_backbone),
        }
        out={}
        for name,(online,teacher) in pairs.items():
            numerator=torch.zeros((),device=self.device,dtype=torch.float32)
            denominator=torch.zeros((),device=self.device,dtype=torch.float32)
            for p_online,p_teacher in zip(online.parameters(),teacher.parameters()):
                numerator=numerator+(p_online.detach().float()-p_teacher.detach().float()).square().sum()
                denominator=denominator+p_teacher.detach().float().square().sum()
            out[name]=torch.sqrt(numerator)/torch.sqrt(denominator).clamp_min(1e-12)
        return out

    def _gather_unique_validation_stats(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Return unique-admission validation aggregates across DDP ranks.

        Lightning's default validation DistributedSampler may pad indices when
        the split is not divisible by world size. We retain its equal-step
        property, then collapse identical admission records here. A duplicate
        whose deterministic validation statistics disagree is a hard error.
        """
        if self._val_sample_loss_sums:
            local_sums = torch.cat(self._val_sample_loss_sums).detach().float().cpu().tolist()
            local_counts = torch.cat(self._val_sample_counts).detach().float().cpu().tolist()
            local_cosines = torch.cat(self._val_sample_cosine_sums).detach().float().cpu().tolist()
        else:
            local_sums, local_counts, local_cosines = [], [], []
        local_payload = list(zip(self._val_sample_ids, local_sums, local_counts, local_cosines))
        payloads: List[List[Tuple[str, float, float, float]]] = [local_payload]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            payloads = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(payloads, local_payload)
        unique: Dict[str, Tuple[float, float, float]] = {}
        for payload in payloads:
            for sample_id, loss_sum, count, cosine_sum in payload:
                value = (float(loss_sum), float(count), float(cosine_sum))
                previous = unique.get(str(sample_id))
                if previous is not None and not all(
                    math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)
                    for a, b in zip(previous, value)
                ):
                    raise RuntimeError(
                        "Validation admission was duplicated across DDP ranks with inconsistent "
                        f"statistics: {sample_id}."
                    )
                unique[str(sample_id)] = value
        total_loss = sum(value[0] for value in unique.values())
        total_count = sum(value[1] for value in unique.values())
        total_cosine = sum(value[2] for value in unique.values())
        return (
            torch.tensor(total_loss, device=self.device),
            torch.tensor(total_count, device=self.device),
            torch.tensor(total_cosine, device=self.device),
            len(unique),
        )

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        out = self._forward_phase1_numeric_jepa(batch, output_mode="telemetry")
        pred_valid = out["loss_inputs"]["pred_valid"]
        target_valid = out["loss_inputs"]["target_valid"]
        valid_mask = out["loss_inputs"]["valid_mask"]
        values = out["packed_true_values"]
        anchor = self._ddp_zero_target_anchor()
        local_loss_sum = F.l1_loss(pred_valid, target_valid, reduction="sum") if pred_valid.numel() else anchor * 0.0
        local_count = torch.tensor(float(pred_valid.numel()), device=self.device)
        loss = local_loss_sum / local_count.clamp_min(1.0)
        local_cosine = F.cosine_similarity(pred_valid, target_valid, dim=-1) if pred_valid.numel() else pred_valid.new_zeros((0,))
        local_cosine_sum = local_cosine.sum() if local_cosine.numel() else anchor * 0.0
        batch_size = len(batch["hadm_id"])
        sample_loss_sums = pred_valid.new_zeros((batch_size,))
        sample_counts = pred_valid.new_zeros((batch_size,))
        sample_cosine_sums = pred_valid.new_zeros((batch_size,))
        packed_batch = out["packed_batch_indices"]
        if packed_batch.numel():
            sample_loss_sums.index_add_(0, packed_batch, (pred_valid - target_valid).abs().sum(dim=-1))
            sample_counts.index_add_(0, packed_batch, torch.full_like(packed_batch, self.embed_dim, dtype=pred_valid.dtype))
            sample_cosine_sums.index_add_(0, packed_batch, local_cosine)
        self._val_sample_ids.extend(str(item) for item in batch["hadm_id"])
        self._val_sample_loss_sums.append(sample_loss_sums.detach())
        self._val_sample_counts.append(sample_counts.detach())
        self._val_sample_cosine_sums.append(sample_cosine_sums.detach())
        # This fits and evaluates Ridge on the same bounded batch sample. It
        # is intentionally optimistic and never participates in checkpoint
        # selection; retain it only as a cheap representation-health probe.
        should_readout = (batch_idx % self.numeric_readout_every_n_val_batches) == 0
        if should_readout and pred_valid.size(0) >= 2:
            sampled_pred = pred_valid.detach()[:self.validation_telemetry_max_tokens]
            sampled_target = target_valid.detach()[:self.validation_telemetry_max_tokens]
            sampled_values = values[valid_mask].detach()[:self.validation_telemetry_max_tokens]
            teacher_r2, _ = _ridge_readout_stats(sampled_target, sampled_values)
            prediction_r2, _ = _ridge_readout_stats(sampled_pred, sampled_values)
            self.log("val/debug_inbatch_teacher_preset_event_value_readout_r2", teacher_r2, on_epoch=True, batch_size=batch_size)
            self.log("val/debug_inbatch_prediction_value_readout_r2", prediction_r2, on_epoch=True, batch_size=batch_size)
        context = out["student_z_context"]
        seq_pad = batch["seq_pad_mask"].bool()
        context_valid = context[~seq_pad][:self.validation_telemetry_max_tokens]
        self._val_loss_sum = self._val_loss_sum + local_loss_sum.detach()
        self._val_masked_count = self._val_masked_count + local_count.detach()
        self._val_cosine_sum = self._val_cosine_sum + local_cosine_sum.detach()
        self._val_local_norm_sum = self._val_local_norm_sum + out["local_z_set_norm_sum"].detach()
        self._val_local_norm_count = self._val_local_norm_count + out["local_z_set_norm_count"].float().detach()
        if should_readout:
            zset_valid = out["student_z_set"][~seq_pad][:self.validation_telemetry_max_tokens]
            remaining = max(0, self.validation_telemetry_max_tokens - sum(x.size(0) for x in self._val_context_samples))
            if remaining:
                self._val_context_samples.append(context_valid.detach()[:remaining])
                self._val_zset_samples.append(zset_valid.detach()[:remaining])
        if (batch_idx % self.validation_value_intervention_every_n_val_batches) == 0:
            self._collect_validation_value_intervention(batch, out)
        return loss

    def on_validation_epoch_end(self) -> None:
        unique_loss_sum, unique_count, unique_cosine_sum, unique_admissions = self._gather_unique_validation_stats()
        count = unique_count.clamp_min(1.0)
        current_val_loss = unique_loss_sum / count
        local_context = torch.cat(self._val_context_samples, dim=0) if self._val_context_samples else torch.zeros((0, self.embed_dim), device=self.device)
        local_zset = torch.cat(self._val_zset_samples, dim=0) if self._val_zset_samples else torch.zeros((0, self.embed_dim), device=self.device)
        global_context = self._gather_global_timestamp_sample(local_context)
        global_zset = self._gather_global_timestamp_sample(local_zset)
        current_val_rank_eff = effective_rank(global_context) if global_context.size(0) > 10 else 0.0
        self._current_val_loss_for_callbacks = current_val_loss.detach()
        self._current_val_rank_eff_for_callbacks = current_val_rank_eff
        self.log("val/loss", current_val_loss, prog_bar=True, logger=True, sync_dist=False, batch_size=1)
        self.log("val/loss_sum", unique_loss_sum, sync_dist=False, batch_size=1)
        masked_events = count / float(self.embed_dim)
        self.log("val/feature_token_cosine", unique_cosine_sum / masked_events.clamp_min(1.0), sync_dist=False, batch_size=1)
        self.log("val/num_masked_numeric_events", masked_events, sync_dist=False, batch_size=1)
        self.log("val/num_masked_latent_elements", unique_count, sync_dist=False, batch_size=1)
        self.log("val/unique_admissions", float(unique_admissions), sync_dist=False, batch_size=1)
        self.log("val/num_masked_categorical", torch.zeros((), device=self.device), sync_dist=False, batch_size=1)
        global_local_norm_sum = self._distributed_sum(self._val_local_norm_sum)
        global_local_norm_count = self._distributed_sum(self._val_local_norm_count)
        self.log("val/local_z_set_norm", global_local_norm_sum / global_local_norm_count.clamp_min(1.0), sync_dist=False, batch_size=1)
        self.log("val_rank_eff", current_val_rank_eff, sync_dist=False, batch_size=1)
        self.log("val/z_set_rank_eff", effective_rank(global_zset) if global_zset.size(0) > 10 else 0.0, sync_dist=False, batch_size=1)
        for name,value in self._teacher_online_parameter_distance().items():
            self.log(f"val/teacher_online_{name}_relative_parameter_distance", value, sync_dist=False, batch_size=1)
        intervention=self._gather_value_intervention_records()
        if intervention:
            for name in (
                "teacher_preset_event_delta_norm", "teacher_set_member_delta_norm", "teacher_set_member_over_preset_event",
                "online_projector_delta_norm", "online_set_member_delta_norm", "online_set_member_over_projector",
            ):
                self.log(
                    f"val/value_intervention_{name}",
                    float(np.mean([float(row[name]) for row in intervention])),
                    sync_dist=False,batch_size=1,
                )
            self.log("val/value_intervention_n_unique_samples",float(len(intervention)),sync_dist=False,batch_size=1)

    @torch.no_grad()
    def extract_retrospective_teacher_sequence(
        self,
        batch: Dict[str, torch.Tensor],
        task_cutoff_min: float,
        base_representations: Optional[List[str]] = None,
        include_teacher_z_context: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Retrospective downstream extraction from Phase-1 path.

        Main output remains teacher z_context sequence (B,T,D):
          EMA time-free front-end -> public EMA context helper with outer z_set skip.

        Optional base representations for downstream ablations:
          - student: online state_backbone output
          - visit: SetEncoder visit embedding before temporal backbone
        """
        state_tokens = batch["state_tokens"]
        state_values = batch["state_values"]
        s_pad = batch["state_pad_mask"]
        seq_pad = batch["seq_pad_mask"]
        time = batch["state_times"]
        B, T, S_s = state_tokens.shape
        D = self.embed_dim

        teacher_z_context: Optional[torch.Tensor] = None
        if include_teacher_z_context:
            _, z_visit, teacher_z_context = self._encode_teacher_states(
                state_tokens, state_values, s_pad, seq_pad, time
            )
        else:
            fp, fnorm, senc = self._teacher_frontend_modules()
            z_s_clean_norm = self._encode_event_tokens(
                state_tokens, state_values, projector=fp, fusion_norm=fnorm
            ).masked_fill(s_pad.unsqueeze(-1), 0.0)
            z_visit = senc(
                z_s_clean_norm.view(B * T, S_s, D),
                s_pad.view(B * T, S_s),
            ).view(B, T, D)

        requested = set(str(x) for x in (base_representations or []))
        z_student_bidir: Optional[torch.Tensor] = None
        if "student" in requested:
            z_s_norm_student = self._encode_event_tokens(state_tokens, state_values)
            z_visit_student = self.set_encoder(
                z_s_norm_student.view(B * T, S_s, D),
                s_pad.view(B * T, S_s),
            ).view(B, T, D)
            student_delta = self.state_backbone(z_visit_student, seq_pad, time, causal_mask=None)
            z_student_bidir = self.context_output_norm(z_visit_student + student_delta)
            z_student_bidir = z_student_bidir.masked_fill(seq_pad.unsqueeze(-1), 0.0)

        abs_time = time.squeeze(-1) if time.dim() == 3 else time
        has_observation = (~s_pad).any(dim=-1) & (~seq_pad)

        if "study_start" not in batch:
            raise RuntimeError(
                "Retrospective teacher extraction requires batch['study_start'] for cutoff-safe t_post."
            )
        study_start_t = batch["study_start"].float().to(abs_time.device)
        if study_start_t.dim() == 1:
            study_start_t = study_start_t.unsqueeze(-1)
        if study_start_t.dim() != 2 or study_start_t.size(-1) != 1:
            raise RuntimeError(
                f"Unexpected study_start shape: {tuple(study_start_t.shape)}; expected (B,1) or (B,)."
            )
        t_post = abs_time - study_start_t

        valid_window = (~seq_pad) & (t_post >= 0.0)
        valid_window = valid_window & (t_post <= float(task_cutoff_min))

        out: Dict[str, torch.Tensor] = {
            "seq_pad": seq_pad,
            "seq_pad_window": ~valid_window,
            "abs_time": abs_time,
            "t_post": t_post,
            "has_observation": has_observation,
            "has_observation_window": has_observation & valid_window,
            "valid_window": valid_window,
        }
        if teacher_z_context is not None:
            teacher_z_context_window = teacher_z_context.masked_fill(~valid_window.unsqueeze(-1), 0.0)
            out["teacher_z_context"] = teacher_z_context_window
        if z_student_bidir is not None:
            out["z_student_bidir"] = z_student_bidir.masked_fill(~valid_window.unsqueeze(-1), 0.0)
        if "visit" in requested:
            out["visit_embedding"] = z_visit.masked_fill(~valid_window.unsqueeze(-1), 0.0)
        return out


    def configure_optimizers(self) -> dict:
        # Exclude full EMA teacher params: updated by EMA only.
        target_modules = [
            self.target_feature_projector,
            self.target_fusion_norm,
            self.target_set_encoder,
            self.target_backbone,
            self.target_context_output_norm,
        ]
        target_ids = {id(p) for m in target_modules for p in m.parameters()}
        online_params = [p for p in self.parameters() if id(p) not in target_ids]
        optimizer = AdamW(online_params, lr=self.max_lr, weight_decay=self.wd)
        
        # Scheduler Configuration
        scheduler_cfg = getattr(self, "phase1_config", {}).get("lr_scheduler", {})
        if scheduler_cfg.get("type") == "cosine_with_warmup":
            warmup_steps = scheduler_cfg.get("warmup_steps", 2000)
            min_lr = scheduler_cfg.get("min_lr", 1e-6)
            max_lr = self.max_lr
            
            # Use Lightning's estimated steps if available
            try:
                total_steps = self.trainer.estimated_stepping_batches
            except:
                # Fallback for manual or early init
                total_steps = 25 * 1000 # Default estimate
                
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    # Linear Warmup
                    return min_lr / max_lr + (1.0 - min_lr / max_lr) * (current_step / warmup_steps)
                # Cosine Decay
                progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
                cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
                return min_lr / max_lr + (1.0 - min_lr / max_lr) * cosine
                
            scheduler = LambdaLR(optimizer, lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }
        
        return optimizer

    def _gradient_norms(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Return finite L2 norms and the number of non-finite grad elements."""
        prefixes = {
            "feature_projector": ("feature_projector.",),
            "fusion_norm": ("fusion_norm.",),
            "set_encoder": ("set_encoder.",),
            "backbone": ("state_backbone.", "context_output_norm."),
            "query_predictor": ("z_constructor.", "predictor."),
        }
        total_sq = torch.zeros((), device=self.device, dtype=torch.float32)
        nonfinite = torch.zeros((), device=self.device, dtype=torch.float32)
        grouped_sq = {name: torch.zeros((), device=self.device, dtype=torch.float32) for name in prefixes}
        for name, parameter in self.named_parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach().float()
            safe_grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            squared = (safe_grad * safe_grad).sum()
            total_sq = total_sq + squared
            nonfinite = nonfinite + (~torch.isfinite(grad)).sum().to(torch.float32)
            for group, group_prefixes in prefixes.items():
                if name.startswith(group_prefixes):
                    grouped_sq[group] = grouped_sq[group] + squared
                    break
        return torch.sqrt(total_sq), {name: torch.sqrt(value) for name, value in grouped_sq.items()}, nonfinite

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        """Normalize one accumulation cycle and invoke Lightning clipping once."""
        cycle_count = self._phase1_cycle_global_count
        denominator = cycle_count.clamp_min(1.0) if cycle_count is not None else torch.ones((), device=self.device)
        emit = int(getattr(self, "global_step", 0)) % self.train_telemetry_every_n_steps == 0
        raw_norm, _, raw_nonfinite = self._gradient_norms() if emit else (None, None, None)
        for parameter in self.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(denominator)
        normalized_norm, group_norms, normalized_nonfinite = self._gradient_norms() if emit else (None, None, None)
        if gradient_clip_val is not None and float(gradient_clip_val) > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=float(gradient_clip_val),
                gradient_clip_algorithm=gradient_clip_algorithm or "norm",
            )
        post_clip_norm, _, post_clip_nonfinite = self._gradient_norms() if emit else (None, None, None)
        if emit:
            cycle_events = self._phase1_cycle_masked_events
            if cycle_events is None:
                cycle_events = torch.zeros((), device=self.device)
            metrics = {
                "grad/raw_sum_global_norm": raw_norm,
                "grad/normalized_global_norm": normalized_norm,
                "grad/post_clip_global_norm": post_clip_norm,
                "grad/clip_ratio": post_clip_norm / normalized_norm.clamp_min(torch.finfo(torch.float32).eps),
                "grad/nonfinite_gradient_elements": self._distributed_sum(raw_nonfinite + normalized_nonfinite + post_clip_nonfinite),
                "grad/cycle_masked_numeric_events": cycle_events,
                "grad/cycle_masked_latent_elements": denominator,
                "grad/cycle_normalization_denominator": denominator,
            }
            metrics.update({f"grad/normalized_{name}_norm": value for name, value in group_norms.items()})
            for name, value in metrics.items():
                self.log(name, value, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=1)
        self._phase1_cycle_global_count = None
        self._phase1_cycle_masked_events = None

    def _update_phase1_ema_once(self) -> None:
        """Update every frozen teacher module after one completed optimizer step."""
        self.ema_feature_projector_updater.update()
        self.ema_fusion_norm_updater.update()
        self.ema_set_encoder_updater.update()
        self.ema_backbone_updater.update()
        self.ema_context_output_norm_updater.update()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None) -> None:
        """Run EMA strictly after Lightning has applied the online optimizer step."""
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self._update_phase1_ema_once()
