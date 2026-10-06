import os
import hashlib
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from functools import lru_cache
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List


def get_autocast_dtype(device_type: str) -> torch.dtype:
    """
    Returns the best available autocast dtype for the given device.
    Falls back to float16 if bfloat16 is not supported on CUDA.
    """
    if device_type == "cpu":
        return torch.bfloat16
    if torch.cuda.is_available() and device_type == "cuda":
        # Check for bfloat16 support (Ampere+ GPUs)
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float16  # Default fallback for CUDA or other accelerators


# Constants
DEFAULT_CONFIG_PATH = Path("configs/params.yaml")
PHASE1_MASKING_CONFIG_KEYS = frozenset({
    "mask_target_ratio", "mask_rarity_strength", "mask_min_prob", "mask_max_prob",
    "static_mask_prob", "static_numeric_vids",
})
TASK_TO_LABEL_COL = {
    "mortality": "label_mortality",
    "readmission": "label_readmission_30d",
    "los": "label_los_days",
}

# Shared cache layout (single source of truth).

class ConfigError(Exception):
    """Configuration related error."""
    pass


def task_cutoff_min(task: str) -> float:
    """Canonical retrospective cutoff (minutes) per downstream task."""
    return 1440.0 if str(task).lower() == "los" else 2880.0




def benchmark_cache_root(cache_dir: str) -> str:
    """Shared benchmark cache root."""
    return os.path.join(cache_dir, "benchmark_features")


def benchmark_metadata_path(cache_dir: str) -> str:
    """Metadata path for shared benchmark cache."""
    return os.path.join(benchmark_cache_root(cache_dir), "metadata.json")




def set_seed(seed: int) -> None:
    """Set python / numpy / torch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    if y_true.size == 0:
        return 0.0
    y_prob = np.clip(y_prob.astype(np.float64), 1e-7, 1.0 - 1e-7)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece = 0.0
    n = float(y_true.shape[0])
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        acc = float(np.mean(y_true[m]))
        conf = float(np.mean(y_prob[m]))
        ece += abs(acc - conf) * (float(np.sum(m)) / n)
    return float(ece)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Shared binary classification metrics.
    Returns defaults for empty or single-class targets to keep downstream code stable.
    """
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float32).reshape(-1)
    if y_true.size == 0 or y_prob.size == 0 or y_true.size != y_prob.size:
        return {"auprc": 0.5, "auroc": 0.5, "brier": 0.25, "ece": 0.0, "n": int(y_true.size)}

    y_prob = np.clip(y_prob, 1e-7, 1.0 - 1e-7)
    brier = float(np.mean((y_prob - y_true) ** 2))
    ece = expected_calibration_error(y_true, y_prob)

    if len(np.unique(y_true)) < 2:
        return {"auprc": 0.5, "auroc": 0.5, "brier": brier, "ece": ece, "n": int(y_true.size)}

    from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

    p, r, _ = precision_recall_curve(y_true, y_prob)
    return {
        "auprc": float(auc(r, p)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "brier": brier,
        "ece": ece,
        "n": int(y_true.size),
    }









@dataclass
class RecipeConfig:
    """
    High-level 'Chef's Recipe' for JEPAModel.
    Controls architectural choices via semantic switches.
    """
    # 1. Set Encoder (Intra-visit token-mixer depth)
    set_layers: int = 2

    # 2. Sequence Backbone (Inter-visit)
    use_delta_t: bool = True
    context_layers: int = 4
    predictor_layers: int = 8  # Explicit predictor depth
    
    # 3. Phase-1 numeric frequency-guided masking
    mask_target_ratio: float = 0.15
    mask_rarity_strength: float = 0.40
    mask_min_prob: float = 0.05
    mask_max_prob: float = 0.50
    static_mask_prob: float = 0.15
    static_numeric_vids: Tuple[int, ...] = (5921,)
    
    # 4. Stability
    ema_decay: float = 0.996


class MaskedLayerNorm(nn.Module):
    """
    LayerNorm that strictly ignores and zeroes out padding positions.
    Output satisfy: norm(z[padding]) == 0.
    """
    def __init__(self, embed_dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (..., D)
        mask: (..., ) Bool, True=Padding
        """
        valid_mask = (~mask).unsqueeze(-1).float()
        
        # Sanitize x to prevent NaN propagation from masked attention
        if x.isnan().any():
            x = torch.nan_to_num(x, nan=0.0)
        
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mu) / torch.sqrt(var + self.eps)
        
        out = x_norm * self.gamma + self.beta
        return out * valid_mask

def validate_phase1_masking_config(raw: Dict[str, Any]) -> None:
    """Reject removed and unknown Phase-1 masking settings before dataset construction."""
    phase1 = raw.get("phase1", {}) if isinstance(raw, dict) else {}
    found = sorted(
        key
        for section in ("data", "recipe", "normalization", "training")
        if isinstance(phase1.get(section), dict)
        for key in phase1[section]
        if (key.startswith("mask_") and key not in PHASE1_MASKING_CONFIG_KEYS) or key.startswith("block_")
    )
    if found:
        raise ConfigError(
            "Unsupported Phase-1 masking configuration keys: " + ", ".join(found)
        )


def load_config_dict(config_path: "str | Path" = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration from YAML."""
    resolved_path = Path(config_path).expanduser().resolve()
    if not resolved_path.exists():
        return {}

    try:
        with resolved_path.open("r", encoding="utf-8") as fp:
            raw = yaml.safe_load(fp) or {}
    except Exception as e:
        raise ConfigError(f"Failed to load config at {resolved_path}: {e}")

    validate_phase1_masking_config(raw)
    return raw

def effective_rank(z: torch.Tensor, epsilon: float = 1e-12) -> float:
    """
    Computes Effective Rank (Shannon Entropy of Singular Values).
    """
    if z.size(0) < 2:
        return 0.0
    
    z = z.detach().float()
    z_centered = z - z.mean(dim=0, keepdim=True)
    
    s = torch.linalg.svdvals(z_centered)
    
    eig_vals = (s ** 2) / (z.size(0) - 1)
    p = eig_vals / (eig_vals.sum() + epsilon)
    entropy = -(p * torch.log(p + epsilon)).sum()
    
    return torch.exp(entropy).item()

def safe_attention(q, k, v, mask=None, score_bias=None, dropout_p: float = 0.0,
                   return_attn: bool = False) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor]":
    """
    Manual Scaled Dot Product Attention with Numerical Safety Guards.
    """
    q = torch.clamp(q, min=-_ATTN_QKV_CLAMP, max=_ATTN_QKV_CLAMP)
    k = torch.clamp(k, min=-_ATTN_QKV_CLAMP, max=_ATTN_QKV_CLAMP)
    v = torch.clamp(v, min=-_ATTN_QKV_CLAMP, max=_ATTN_QKV_CLAMP)

    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("safe_attention expects q, k and v with shape (B, H, L, D).")
    score_shape = (q.size(0), q.size(1), q.size(2), k.size(2))

    def _additive_scores(x, *, name: str, boolean_means_mask: bool) -> torch.Tensor:
        if x.dim() > 4:
            raise ValueError(f"{name} must have at most four dimensions, got {tuple(x.shape)}")
        while x.dim() < 4:
            x = x.unsqueeze(0)
        try:
            torch.broadcast_shapes(tuple(x.shape), score_shape)
        except RuntimeError as exc:
            raise ValueError(
                f"{name} shape {tuple(x.shape)} is not broadcastable to attention scores {score_shape}"
            ) from exc
        if boolean_means_mask and x.dtype == torch.bool:
            additive = torch.zeros_like(x, dtype=q.dtype)
            return additive.masked_fill(x, float("-inf"))
        additive = x.to(q.dtype)
        finite_safe = torch.nan_to_num(additive, nan=0.0, posinf=1e4, neginf=-1e4)
        return torch.where(additive == float("-inf"), additive, finite_safe)

    additive_mask = _additive_scores(mask, name="mask", boolean_means_mask=True) if mask is not None else None
    additive_bias = _additive_scores(score_bias, name="score_bias", boolean_means_mask=False) if score_bias is not None else None

    if not return_attn:
        # Chunked manual attention: peak memory O(chunk * L), no SDPA backend issues.
        Q = q.size(2)
        CHUNK = _ATTN_CHUNK_SIZE
        d_k_fast = q.size(-1)
        inv_sqrt = 1.0 / math.sqrt(d_k_fast)

        # Pre-process mask and optional additive score bias once.
        fast_mask = additive_mask
        fast_bias = additive_bias

        out_chunks = []
        for i in range(0, Q, CHUNK):
            q_c = q[:, :, i:i + CHUNK]                                         # (B,H,c,D)
            if fast_mask is not None:
                if fast_mask.size(2) == 1:
                    # Padding-only masks often carry a singleton query axis.
                    # Broadcast that axis across the current chunk instead of slicing it away.
                    m_c = fast_mask[:, :, :1, :].expand(-1, -1, q_c.size(2), -1)
                else:
                    m_c = fast_mask[:, :, i:i + q_c.size(2), :]
                if m_c.size(2) != q_c.size(2):
                    raise RuntimeError(
                        f"safe_attention chunk mask mismatch: q_chunk={tuple(q_c.shape)}, "
                        f"mask_chunk={tuple(m_c.shape)}, full_mask={tuple(fast_mask.shape)}"
                    )
            else:
                m_c = None
            if fast_bias is not None:
                if fast_bias.size(2) == 1:
                    b_c = fast_bias[:, :, :1, :].expand(-1, -1, q_c.size(2), -1)
                else:
                    b_c = fast_bias[:, :, i:i + q_c.size(2), :]
                if b_c.size(2) != q_c.size(2):
                    raise RuntimeError(
                        f"safe_attention chunk bias mismatch: q_chunk={tuple(q_c.shape)}, "
                        f"bias_chunk={tuple(b_c.shape)}, full_bias={tuple(fast_bias.shape)}"
                    )
            else:
                b_c = None
            sc = torch.matmul(q_c, k.transpose(-2, -1)) * inv_sqrt             # (B,H,c,L)
            sc = torch.clamp(sc, -_ATTN_SCORE_CLAMP, _ATTN_SCORE_CLAMP)
            if b_c is not None:
                sc = sc + b_c
            if m_c is not None:
                sc = sc + m_c
            all_m = (sc == float('-inf')).all(dim=-1, keepdim=True)
            sc = sc.masked_fill(all_m, 0.0)
            p = torch.softmax(sc.float(), dim=-1).to(q.dtype)
            if m_c is not None:
                p = p.masked_fill(m_c == float('-inf'), 0.0)
            p = p.masked_fill(all_m, 0.0)
            if dropout_p > 0:
                p = F.dropout(p, p=dropout_p)
            out_chunks.append(torch.matmul(p, v))
        return torch.nan_to_num(torch.cat(out_chunks, dim=2), nan=0.0)

    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    scores = torch.clamp(scores, min=-_ATTN_SCORE_CLAMP, max=_ATTN_SCORE_CLAMP)
    if additive_bias is not None:
        scores = scores + additive_bias

    # Track which query rows have ALL KV positions masked (fully-blocked rows).
    # For these rows, softmax would produce 0/0 = NaN.
    all_masked = None
    if additive_mask is not None:
        mask = additive_mask
        scores = scores + additive_mask
        # Detect fully-masked rows: all positions are -inf after masking
        all_masked = (scores == float('-inf')).all(dim=-1, keepdim=True)  # (..., 1)
        # Replace scores for fully-masked rows with 0 so softmax returns
        # uniform 1/L (well-defined, gradient-safe) instead of NaN.
        scores = scores.masked_fill(all_masked, 0.0)

    max_scores = scores.max(dim=-1, keepdim=True)[0].detach()
    scores = scores - max_scores
    p_attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)

    # [STRICT CAUSAL GUARD] Zero out attention to masked positions.
    if mask is not None:
        p_attn = p_attn.masked_fill(mask == float('-inf'), 0.0)

    # Zero out entire rows that were fully masked (attend to nothing).
    if all_masked is not None:
        p_attn = p_attn.masked_fill(all_masked, 0.0)
    
    if dropout_p > 0:
        p_attn = F.dropout(p_attn, p=dropout_p)
    
    v = torch.clamp(v, min=-_ATTN_QKV_CLAMP, max=_ATTN_QKV_CLAMP)
    output = torch.matmul(p_attn, v)
    
    output = torch.nan_to_num(output, nan=0.0)
        
    if return_attn:
        return output, p_attn
    return output

class ResilientTransformerLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.norm1 = MaskedLayerNorm(d_model)
        self.norm2 = MaskedLayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*4, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor, causal_mask: Optional[torch.Tensor] = None, return_attn=False) -> torch.Tensor:
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        
        combined_mask = None
        if mask is not None:
            pad_mask = mask.view(B, 1, 1, L)
            
        if causal_mask is not None:
            if isinstance(causal_mask, bool):
                c_mask = torch.triu(torch.ones(L, L, device=x.device) * float('-inf'), diagonal=1) if causal_mask else None
            else:
                c_mask = causal_mask
                
            if c_mask is not None:
                c_mask = c_mask.unsqueeze(0).unsqueeze(0)
                if mask is not None:
                    pad_additive = torch.zeros_like(pad_mask, dtype=q.dtype)
                    pad_additive.masked_fill_(pad_mask, float('-inf'))
                    combined_mask = c_mask + pad_additive
                else:
                    combined_mask = c_mask
        elif mask is not None:
             pad_additive = torch.zeros_like(pad_mask, dtype=q.dtype)
             pad_additive.masked_fill_(pad_mask, float('-inf'))
             combined_mask = pad_additive
 
        res = safe_attention(
            q, k, v, mask=combined_mask, 
            dropout_p=self.dropout.p if self.training else 0.0,
            return_attn=return_attn
        )
        if return_attn:
            attn_out, attn_weights = res
        else:
            attn_out = res
            attn_weights = None
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        attn_out = torch.clamp(attn_out, min=-50, max=50)
        
        out = x + self.out_proj(attn_out)
        out = self.norm1(out, mask)
        out = out + self.ff(out)
        out = self.norm2(out, mask)
        
        if return_attn:
            return out, attn_weights
        return out

class EMAMomentumUpdater:
    def __init__(self, target_net: nn.Module, online_net: nn.Module, decay: float = 0.996):
        self.target = target_net
        self.online = online_net
        self.decay = decay
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False
            
    def update(self, decay: float = None) -> None:
        d = decay if decay is not None else self.decay
        with torch.no_grad():
            target_by_dtype = {}
            online_by_dtype = {}
            for p_t, p_o in zip(self.target.parameters(), self.online.parameters()):
                if p_t.device != p_o.device or p_t.dtype != p_o.dtype:
                    p_t.mul_(d).add_(p_o, alpha=1 - d)
                    continue
                key = (p_t.device, p_t.dtype)
                target_by_dtype.setdefault(key, []).append(p_t)
                online_by_dtype.setdefault(key, []).append(p_o)
            for key, targets in target_by_dtype.items():
                torch._foreach_mul_(targets, d)
                torch._foreach_add_(targets, online_by_dtype[key], alpha=1 - d)

class ResilientTransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ResilientTransformerLayer(embed_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        self.gradient_checkpointing = False
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor, causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            import torch.utils.checkpoint as checkpoint
            for layer in self.layers:
                x = checkpoint.checkpoint(layer, x, mask, causal_mask, use_reentrant=False)
        else:
            for layer in self.layers:
                x = layer(x, mask, causal_mask)
        return x

    def forward_with_attn(self, x: torch.Tensor, mask: torch.Tensor, causal_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        attn_list = []
        for layer in self.layers:
            x, attn = layer(x, mask, causal_mask, return_attn=True)
            attn_list.append(attn)
        return x, attn_list

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(t.float())


# Module-level shared numerical constants (loaded from config at import time)
_SHARED_NUM = load_config_dict().get("shared", {}).get("numerical", {})
_ATTN_QKV_CLAMP: float = _SHARED_NUM.get("attention_qkv_clamp", 50.0)
_ATTN_SCORE_CLAMP: float = _SHARED_NUM.get("attention_score_clamp", 20.0)
_ATTN_CHUNK_SIZE: int = int(_SHARED_NUM.get("attention_chunk_size", 512))


def parse_phenotype_code_list(raw: Any) -> List[str]:
    """Parse raw phenotype code strings into deduplicated tokens."""
    if raw is None:
        return []
    if isinstance(raw, float) and np.isnan(raw):
        return []
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    out: List[str] = []
    seen = set()
    for part in text.split(","):
        code = str(part).strip()
        if not code or code in seen:
            continue
        seen.add(code)
        out.append(code)
    return out


@lru_cache(maxsize=16)
def _load_labels_csv_cached(abs_labels_csv: str) -> pd.DataFrame:
    df = pd.read_csv(abs_labels_csv).copy()
    if "label_los_days" not in df.columns:
        if "hospital_los" in df.columns:
            df["label_los_days"] = pd.to_numeric(df["hospital_los"], errors="coerce")
        elif "unit_los" in df.columns:
            df["label_los_days"] = pd.to_numeric(df["unit_los"], errors="coerce")
    if "label_hospital_los" not in df.columns and "hospital_los" in df.columns:
        df["label_hospital_los"] = pd.to_numeric(df["hospital_los"], errors="coerce")
    if "label_unit_los" not in df.columns and "unit_los" in df.columns:
        df["label_unit_los"] = pd.to_numeric(df["unit_los"], errors="coerce")
    return df


def load_labels_csv(labels_csv: str) -> pd.DataFrame:
    return _load_labels_csv_cached(os.path.abspath(str(labels_csv))).copy()


@lru_cache(maxsize=16)
def _load_schema_csv_cached(abs_schema_csv: str) -> pd.DataFrame:
    return pd.read_csv(abs_schema_csv).copy()


def load_schema_csv(schema_csv: str) -> pd.DataFrame:
    return _load_schema_csv_cached(os.path.abspath(str(schema_csv))).copy()




def build_phenotype_vocab_metadata(
    labels_df: pd.DataFrame,
    label_col: str,
    min_code_count: int,
    max_codes: Optional[int],
    include_other_bucket: bool = False,
) -> Dict[str, Any]:
    train_df = labels_df[labels_df["split"].astype(str) == "train"].copy()
    counts: Dict[str, int] = {}
    for codes in train_df["parsed_phenotype_codes"].tolist():
        for code in codes:
            counts[code] = counts.get(code, 0) + 1

    items = [(code, int(cnt)) for code, cnt in counts.items() if int(cnt) >= int(min_code_count)]
    items.sort(key=lambda kv: (-int(kv[1]), str(kv[0])))
    if max_codes is not None and int(max_codes) > 0:
        items = items[: int(max_codes)]

    codes = [str(code) for code, _ in items]
    if include_other_bucket:
        codes.append("__OTHER__")
    code_to_idx = {code: idx for idx, code in enumerate(codes)}

    train_targets = np.zeros((len(train_df), len(codes)), dtype=np.float32)
    code_counts_kept = {code: 0 for code in codes}
    for i, row_codes in enumerate(train_df["parsed_phenotype_codes"].tolist()):
        for code in row_codes:
            if code in code_to_idx:
                idx = code_to_idx[code]
                train_targets[i, idx] = 1.0
                code_counts_kept[code] += 1
            elif include_other_bucket and "__OTHER__" in code_to_idx:
                idx = code_to_idx["__OTHER__"]
                train_targets[i, idx] = 1.0
                code_counts_kept["__OTHER__"] += 1

    n_train = max(int(train_targets.shape[0]), 1)
    pos = train_targets.sum(axis=0)
    neg = float(n_train) - pos
    priors = np.clip(pos / float(n_train), 1e-6, 1.0 - 1e-6)

    return {
        "label_col": str(label_col),
        "codes": codes,
        "code_to_idx": code_to_idx,
        "train_code_counts": {k: int(v) for k, v in code_counts_kept.items()},
        "all_train_code_counts_raw": {str(k): int(v) for k, v in counts.items()},
        "n_train_sequences": int(n_train),
        "train_positive_counts": pos.astype(np.int64).tolist(),
        "train_negative_counts": neg.astype(np.int64).tolist(),
        "train_priors": priors.astype(np.float32).tolist(),
        "min_code_count": int(min_code_count),
        "max_codes": None if max_codes is None else int(max_codes),
        "include_other_bucket": bool(include_other_bucket),
    }


def encode_phenotype_target(
    codes: List[str],
    code_to_idx: Dict[str, int],
    include_other_bucket: bool = False,
) -> torch.Tensor:
    target = torch.zeros((len(code_to_idx),), dtype=torch.float32)
    for code in codes:
        if code in code_to_idx:
            target[int(code_to_idx[code])] = 1.0
        elif include_other_bucket and "__OTHER__" in code_to_idx:
            target[int(code_to_idx["__OTHER__"])] = 1.0
    return target


def _safe_multilabel_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return float("nan")


def _safe_multilabel_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ks: List[int],
    threshold: float = 0.5,
) -> Dict[str, float]:
    if y_true.ndim != 2 or y_prob.ndim != 2 or y_true.shape != y_prob.shape or y_true.size == 0:
        out = {
            "micro_auprc": float("nan"),
            "macro_auprc": float("nan"),
            "micro_auroc": float("nan"),
            "macro_auroc": float("nan"),
            "mean_predicted_labels_per_sample": float("nan"),
        }
        for k in ks:
            out[f"precision_at_{int(k)}"] = float("nan")
            out[f"recall_at_{int(k)}"] = float("nan")
        return out

    micro_auprc = _safe_multilabel_ap(y_true.reshape(-1), y_prob.reshape(-1))
    micro_auroc = _safe_multilabel_auc(y_true.reshape(-1), y_prob.reshape(-1))

    macro_prs: List[float] = []
    macro_rocs: List[float] = []
    for j in range(y_true.shape[1]):
        ap = _safe_multilabel_ap(y_true[:, j], y_prob[:, j])
        auc = _safe_multilabel_auc(y_true[:, j], y_prob[:, j])
        if np.isfinite(ap):
            macro_prs.append(float(ap))
        if np.isfinite(auc):
            macro_rocs.append(float(auc))

    out = {
        "micro_auprc": float(micro_auprc),
        "macro_auprc": float(np.mean(macro_prs)) if macro_prs else float("nan"),
        "micro_auroc": float(micro_auroc),
        "macro_auroc": float(np.mean(macro_rocs)) if macro_rocs else float("nan"),
        "mean_predicted_labels_per_sample": float((y_prob >= float(threshold)).sum(axis=1).mean()),
    }

    for k_raw in ks:
        k = int(min(max(1, int(k_raw)), y_true.shape[1]))
        p_vals: List[float] = []
        r_vals: List[float] = []
        for i in range(y_true.shape[0]):
            top_idx = np.argpartition(y_prob[i], -k)[-k:]
            tp = float(y_true[i, top_idx].sum())
            p_vals.append(tp / float(k))
            positives = float(y_true[i].sum())
            if positives > 0:
                r_vals.append(tp / positives)
        out[f"precision_at_{int(k_raw)}"] = float(np.mean(p_vals)) if p_vals else float("nan")
        out[f"recall_at_{int(k_raw)}"] = float(np.mean(r_vals)) if r_vals else float("nan")
    return out


def apply_per_label_thresholds(
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    yp = np.asarray(y_prob, dtype=np.float32)
    th = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
    return (yp >= th).astype(np.float32)


def fit_per_label_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold_grid: List[float],
    objective: str = "f0.5",
    fallback_threshold: float = 0.5,
    min_pos_for_threshold: int = 5,
) -> Dict[str, Any]:
    yt = np.asarray(y_true, dtype=np.float32)
    yp = np.asarray(y_prob, dtype=np.float32)
    if yt.ndim != 2 or yp.ndim != 2 or yt.shape != yp.shape:
        raise ValueError("y_true and y_prob must have the same 2D shape for threshold fitting.")

    grid = [float(x) for x in threshold_grid]
    if not grid:
        raise ValueError("threshold_grid must not be empty.")
    objective_norm = str(objective).lower()
    if objective_norm == "f0.5":
        beta = 0.5
    elif objective_norm in {"f1", "f1.0"}:
        beta = 1.0
    else:
        raise ValueError(f"Unsupported threshold objective: {objective}")
    beta2 = beta * beta
    n_labels = int(yt.shape[1])
    thresholds = np.full((n_labels,), float(fallback_threshold), dtype=np.float32)
    threshold_source = np.full((n_labels,), "fallback", dtype=object)
    best_scores = np.full((n_labels,), np.nan, dtype=np.float32)
    positives = yt.sum(axis=0)

    for j in range(n_labels):
        y = yt[:, j]
        p = yp[:, j]
        if int(positives[j]) < int(min_pos_for_threshold) or len(np.unique(y)) < 2:
            continue
        best_t = float(fallback_threshold)
        best_s = -1.0
        best_precision = -1.0
        best_pred_rate = float("inf")
        for t in grid:
            pred = (p >= float(t)).astype(np.float32)
            tp = float((pred * y).sum())
            fp = float((pred * (1.0 - y)).sum())
            fn = float(((1.0 - pred) * y).sum())
            precision = tp / max(tp + fp, 1e-8)
            recall = tp / max(tp + fn, 1e-8)
            denom = beta2 * precision + recall
            score = 0.0 if denom <= 0.0 else ((1.0 + beta2) * precision * recall / denom)
            pred_rate = float(pred.mean())
            if (
                score > best_s + 1e-12
                or (
                    abs(score - best_s) <= 1e-12
                    and (
                        precision > best_precision + 1e-12
                        or (
                            abs(precision - best_precision) <= 1e-12
                            and (
                                pred_rate < best_pred_rate - 1e-12
                                or (
                                    abs(pred_rate - best_pred_rate) <= 1e-12
                                    and float(t) > best_t
                                )
                            )
                        )
                    )
                )
            ):
                best_s = score
                best_t = float(t)
                best_precision = precision
                best_pred_rate = pred_rate
        thresholds[j] = float(best_t)
        best_scores[j] = float(best_s)
        threshold_source[j] = objective_norm

    return {
        "thresholds": thresholds,
        "scores": best_scores,
        "positive_counts": positives.astype(np.int64),
        "sources": threshold_source,
        "objective": objective_norm,
        "fallback_threshold": float(fallback_threshold),
        "min_pos_for_threshold": int(min_pos_for_threshold),
    }


def summarize_prefix_consistency(
    full_prob: np.ndarray,
    prefix_prob: np.ndarray,
    top_k: int,
) -> Dict[str, float]:
    full = np.asarray(full_prob, dtype=np.float32)
    pref = np.asarray(prefix_prob, dtype=np.float32)
    if full.ndim != 2 or pref.ndim != 2 or full.shape != pref.shape or full.size == 0:
        return {
            "topk_overlap": float("nan"),
            "topk_prob_delta": float("nan"),
            "top1_stability": float("nan"),
        }

    k = int(min(max(1, int(top_k)), full.shape[1]))
    overlaps: List[float] = []
    deltas: List[float] = []
    top1_same: List[float] = []
    for i in range(full.shape[0]):
        top_full = np.argpartition(full[i], -k)[-k:]
        top_pref = np.argpartition(pref[i], -k)[-k:]
        overlap = len(set(top_full.tolist()) & set(top_pref.tolist())) / float(k)
        overlaps.append(float(overlap))

        top_full_sorted = top_full[np.argsort(-full[i, top_full])]
        deltas.append(float(np.mean(np.abs(full[i, top_full_sorted] - pref[i, top_full_sorted]))))
        top1_same.append(float(int(np.argmax(full[i]) == np.argmax(pref[i]))))

    return {
        "topk_overlap": float(np.mean(overlaps)) if overlaps else float("nan"),
        "topk_prob_delta": float(np.mean(deltas)) if deltas else float("nan"),
        "top1_stability": float(np.mean(top1_same)) if top1_same else float("nan"),
    }
