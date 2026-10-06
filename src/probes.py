"""Exact-time downstream probes over frozen Phase-1 sequence caches.

This module intentionally owns the only probe architecture.  Cache inputs use
the public canonical contract: ``z_set``, ``z_context``, ``time_min`` and
``seq_pad_mask``.  Old Phase-1 internal names are not accepted here.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import time
import csv
import subprocess
import sys
import random
from contextlib import nullcontext
from functools import partial
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm

from src.cache_bank import FULL_VISIBLE, RETROSPECTIVE_CUTOFF, load_dataset
from src.model import RelativeTimeTransformerLayer
from src.utils import (TASK_TO_LABEL_COL, TimeEmbedding, build_phenotype_vocab_metadata,
                       compute_binary_metrics, compute_multilabel_metrics,
                       encode_phenotype_target, load_labels_csv, parse_phenotype_code_list)


PROBE_ARCHITECTURE_VERSION = "adaptive_exact_time_probe_v1"
CANONICAL_LATENT_SOURCES = frozenset({"z_set", "z_context"})
TASK_CACHE_KIND = {
    "mortality": (RETROSPECTIVE_CUTOFF, 2880),
    "los_ge_3d": (RETROSPECTIVE_CUTOFF, 1440),
    "los_ge_7d": (RETROSPECTIVE_CUTOFF, 1440),
    "readmission": (FULL_VISIBLE, None),
    "icd": (FULL_VISIBLE, None),
    "hcup": (FULL_VISIBLE, None),
    "drg": (FULL_VISIBLE, None),
}
FOCAL_LOSS_PARAMETERS = {
    "mortality": {"alpha": 0.75, "gamma": 2.0},
    "los_ge_7d": {"alpha": 0.70, "gamma": 2.0},
    "readmission": {"alpha": 0.65, "gamma": 2.0},
}


def _validate_probe_inputs(
    latent_seq: torch.Tensor,
    time_min: torch.Tensor,
    delta_time_min: torch.Tensor,
    seq_pad_mask: torch.Tensor,
) -> None:
    """Validate the exact-time contract without altering invalid examples."""
    if latent_seq.ndim != 3:
        raise ValueError("latent_seq must have shape [B,T,D].")
    b, t, _ = latent_seq.shape
    if t < 1 or time_min.shape != (b, t) or delta_time_min.shape != (b, t):
        raise ValueError("time_min and delta_time_min must have shape [B,T] with T>=1.")
    if seq_pad_mask.shape != (b, t) or seq_pad_mask.dtype != torch.bool:
        raise ValueError("seq_pad_mask must be BoolTensor[B,T].")
    valid = ~seq_pad_mask
    if not valid.any(dim=1).all():
        raise ValueError("Every admission must contain at least one valid timestamp.")
    if not torch.isfinite(latent_seq[valid]).all() or not torch.isfinite(time_min[valid]).all():
        raise ValueError("Valid latent and time values must be finite.")
    if (latent_seq[seq_pad_mask] != 0).any() or (time_min[seq_pad_mask] != 0).any() or (delta_time_min[seq_pad_mask] != 0).any():
        raise ValueError("Padding positions must be zero.")
    if (valid[:, 1:] & ~valid[:, :-1]).any():
        raise ValueError("seq_pad_mask must contain trailing padding only.")
    adjacent = valid[:, 1:] & valid[:, :-1]
    step = time_min[:, 1:] - time_min[:, :-1]
    if not torch.all(step[adjacent] > 0):
        raise ValueError("Valid timestamps must be strictly increasing.")
    expected_delta = torch.zeros_like(delta_time_min)
    expected_delta[:, 1:] = torch.where(adjacent, step, torch.zeros_like(step))
    if not torch.allclose(delta_time_min, expected_delta, atol=1e-5, rtol=1e-5):
        raise ValueError("delta_time_min must be adjacent valid timestamp differences.")


def build_delta_time_min(time_min: torch.Tensor, seq_pad_mask: torch.Tensor, validate: bool = True) -> torch.Tensor:
    if time_min.ndim != 2 or seq_pad_mask.shape != time_min.shape:
        raise ValueError("time_min and seq_pad_mask must both be [B,T].")
    valid = ~seq_pad_mask
    if validate and (not valid.any(dim=1).all() or (valid[:, 1:] & ~valid[:, :-1]).any()):
        raise ValueError("Every admission must have a nonempty right-padded sequence.")
    if validate and not torch.isfinite(time_min[valid]).all():
        raise ValueError("time_min must be finite.")
    delta = torch.zeros_like(time_min)
    adjacent = valid[:, 1:] & valid[:, :-1]
    step = time_min[:, 1:] - time_min[:, :-1]
    if validate and not torch.all(step[adjacent] > 0):
        raise ValueError("time_min must be strictly increasing.")
    delta[:, 1:] = torch.where(adjacent, step, torch.zeros_like(step))
    return delta


class CanonicalProbeDataset(Dataset):
    """Dataset adapter for canonical cache records; it never repairs records."""

    def __init__(self, records: Dataset, source_name: str, targets: Mapping[str, Any], indices: Optional[Sequence[int]] = None):
        if source_name not in CANONICAL_LATENT_SOURCES:
            raise ValueError(f"source_name must be one of {sorted(CANONICAL_LATENT_SOURCES)}.")
        self.records = records
        self.indices = list(range(len(records))) if indices is None else [int(i) for i in indices]
        self.source_name = source_name
        self.targets = dict(targets)
        if not self.indices:
            raise ValueError("Probe dataset cannot be empty.")

    def __len__(self) -> int:
        return len(self.indices)

    def shard_indices(self) -> List[int]:
        if hasattr(self.records, "shard_index"):
            return [int(self.records.shard_index(i)) for i in self.indices]
        return [0] * len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = (self.records.sequence_record(self.indices[index], self.source_name)
                  if hasattr(self.records, "sequence_record") else self.records[self.indices[index]])
        required = {self.source_name, "time_min", "seq_len", "hadm_id"}
        missing = required.difference(record)
        if missing:
            raise RuntimeError(f"Cache record lacks canonical fields: {sorted(missing)}")
        hadm_id = str(record["hadm_id"])
        if hadm_id not in self.targets:
            raise KeyError(f"No target for admission {hadm_id}.")
        # Keep the mmap-backed FP16 latent until batch collation.  Converting
        # here and again in collation doubles CPU allocation and bandwidth.
        z = torch.as_tensor(record[self.source_name])
        time = torch.as_tensor(record["time_min"]).float().reshape(-1)
        if z.ndim != 2 or z.size(0) != time.numel() or z.size(0) != int(record["seq_len"]):
            raise ValueError(f"Invalid canonical sequence for {hadm_id}.")
        if z.size(0) < 1:
            raise ValueError(f"Empty canonical sequence for {hadm_id}.")
        # Collation checks finite values and time order once over the actual
        # padded batch before it can reach the GPU.
        return {"latent_seq": z, "time_min": time, "target": self.targets[hadm_id], "hadm_id": hadm_id}


def canonical_probe_collate(batch: Sequence[Mapping[str, Any]], validate: bool = True) -> Dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty probe batch.")
    d = int(batch[0]["latent_seq"].size(-1))
    lengths = torch.tensor([int(x["latent_seq"].size(0)) for x in batch], dtype=torch.long)
    if (lengths < 1).any() or any(int(x["latent_seq"].size(-1)) != d for x in batch):
        raise ValueError("Probe batch contains an empty sequence or inconsistent latent dimensions.")
    latent = pad_sequence([torch.as_tensor(item["latent_seq"]) for item in batch], batch_first=True, padding_value=0)
    time_min = pad_sequence([torch.as_tensor(item["time_min"], dtype=torch.float32) for item in batch],
                            batch_first=True, padding_value=0.0)
    positions = torch.arange(latent.size(1)).unsqueeze(0)
    pad = positions >= lengths.unsqueeze(1)
    delta = build_delta_time_min(time_min, pad, validate=validate)
    if validate:
        _validate_probe_inputs(latent, time_min, delta, pad)
    raw_targets = [x["target"] for x in batch]
    target = (torch.stack([torch.as_tensor(x) for x in raw_targets])
              if torch.as_tensor(raw_targets[0]).ndim > 0
              else torch.as_tensor(raw_targets))
    return {"latent_seq": latent, "time_min": time_min, "delta_time_min": delta,
            "seq_pad_mask": pad, "lengths": lengths, "target": target,
            "hadm_id": [str(x["hadm_id"]) for x in batch]}


class AttentionCostBatchSampler(Sampler[List[int]]):
    """Length-bucketed batches with a strict pairwise attention budget."""

    def __init__(self, lengths: Sequence[int], max_pairwise_elements_per_batch: int,
                 max_patients_per_batch: int, shuffle: bool, seed: int = 0) -> None:
        if any(int(x) < 1 for x in lengths):
            raise ValueError("All probe sequence lengths must be positive.")
        self.lengths = [int(x) for x in lengths]
        self.limit = int(max_pairwise_elements_per_batch)
        self.max_patients = int(max_patients_per_batch)
        self.shuffle, self.seed, self.epoch = bool(shuffle), int(seed), 0
        self._cached_epoch: Optional[int] = None
        self._cached_batches: Optional[List[List[int]]] = None
        if self.limit < 1 or self.max_patients < 1:
            raise ValueError("Batch limits must be positive.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._cached_epoch = None
        self._cached_batches = None

    def _batches(self) -> List[List[int]]:
        if self._cached_epoch == self.epoch and self._cached_batches is not None:
            return self._cached_batches
        order = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            # Shuffle complete local buckets, avoiding a short-to-long curriculum.
            buckets = [order[i:i + 64] for i in range(0, len(order), 64)]
            rng.shuffle(buckets)
            order = [i for bucket in buckets for i in rng.permutation(bucket)]
        batches: List[List[int]] = []
        current: List[int] = []
        max_length = 0
        for index in order:
            candidate_max = max(max_length, self.lengths[index])
            candidate_cost = (len(current) + 1) * candidate_max ** 2
            if current and (candidate_cost > self.limit or len(current) >= self.max_patients):
                batches.append(current)
                current, max_length = [], 0
            current.append(index)
            max_length = max(max_length, self.lengths[index])
        if current:
            batches.append(current)
        self._cached_epoch, self._cached_batches = self.epoch, batches
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())


class ShardAwareAttentionCostBatchSampler(AttentionCostBatchSampler):
    """Cost-bounded sampler that keeps each epoch local to one cache shard."""

    def __init__(self, lengths: Sequence[int], shard_ids: Sequence[int],
                 max_pairwise_elements_per_batch: int, max_patients_per_batch: int,
                 shuffle: bool, seed: int = 0) -> None:
        if len(lengths) != len(shard_ids):
            raise ValueError("lengths and shard_ids must have equal length.")
        self.shard_ids = [int(x) for x in shard_ids]
        self.shard_switches = 0
        self.batches_per_epoch = 0
        super().__init__(lengths, max_pairwise_elements_per_batch, max_patients_per_batch, shuffle, seed)

    def _batches(self) -> List[List[int]]:
        if self._cached_epoch == self.epoch and self._cached_batches is not None:
            return self._cached_batches
        by_shard: Dict[int, List[int]] = {}
        for index, shard in enumerate(self.shard_ids):
            by_shard.setdefault(shard, []).append(index)
        shard_order = sorted(by_shard)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(shard_order)
        batches: List[List[int]] = []
        for shard in shard_order:
            order = sorted(by_shard[shard], key=lambda i: (self.lengths[i], i))
            if self.shuffle:
                bucket_size = max(16, min(128, self.max_patients * 4))
                buckets = [order[i:i + bucket_size] for i in range(0, len(order), bucket_size)]
                rng.shuffle(buckets)
                order = [i for bucket in buckets for i in rng.permutation(bucket).tolist()]
            current: List[int] = []
            max_length = 0
            for index in order:
                candidate_max = max(max_length, self.lengths[index])
                candidate_cost = (len(current) + 1) * candidate_max ** 2
                if current and (candidate_cost > self.limit or len(current) >= self.max_patients):
                    batches.append(current)
                    current, max_length = [], 0
                current.append(index)
                max_length = max(max_length, self.lengths[index])
            if current:
                batches.append(current)
        self._cached_epoch, self._cached_batches = self.epoch, batches
        self.batches_per_epoch = len(batches)
        self.shard_switches = max(0, len(shard_order) - 1)
        return batches


def thin_probe_batch(batch: Mapping[str, Any], retention_choices: Sequence[float],
                     protected_change_fraction: float, generator: Optional[torch.Generator] = None) -> Dict[str, Any]:
    """Training-only density view retaining endpoints and largest latent changes."""
    if not retention_choices or any(not 0.0 < float(p) <= 1.0 for p in retention_choices):
        raise ValueError("retention_choices must be probabilities in (0,1].")
    source = {k: v for k, v in batch.items()}
    latent, times, pad = source["latent_seq"], source["time_min"], source["seq_pad_mask"]
    b, _, d = latent.shape
    kept: List[torch.Tensor] = []
    for i in range(b):
        n = int((~pad[i]).sum())
        if n < 1:
            raise ValueError("Cannot thin an empty admission.")
        retain = float(retention_choices[int(torch.randint(len(retention_choices), (1,), generator=generator).item())])
        choose = torch.zeros(n, dtype=torch.bool, device=latent.device)
        choose[0] = True
        choose[-1] = True
        if n > 1 and protected_change_fraction > 0:
            changes = torch.linalg.vector_norm(latent[i, 1:n] - latent[i, :n - 1], dim=-1)
            k = min(n - 1, max(1, int(math.ceil((n - 1) * float(protected_change_fraction)))))
            choose[torch.topk(changes, k=k).indices + 1] = True
        random_keep = torch.rand(n, device=latent.device, generator=generator) < retain
        kept.append(torch.where(choose | random_keep)[0])
    max_t = max(int(x.numel()) for x in kept)
    new_latent = latent.new_zeros((b, max_t, d))
    new_times = times.new_zeros((b, max_t))
    new_pad = torch.ones((b, max_t), device=pad.device, dtype=torch.bool)
    for i, ids in enumerate(kept):
        n = int(ids.numel())
        new_latent[i, :n] = latent[i, ids]
        new_times[i, :n] = times[i, ids]
        new_pad[i, :n] = False
    output = dict(source)
    output.update({"latent_seq": new_latent, "time_min": new_times, "seq_pad_mask": new_pad,
                   "delta_time_min": build_delta_time_min(new_times, new_pad),
                   "lengths": (~new_pad).sum(dim=1)})
    return output


class AdaptiveTimeAwareProbeEncoder(nn.Module):
    """Variable-length exact-time encoder with chunked relative-time attention."""

    def __init__(self, embed_dim: int, hidden_dim: int = 128, num_heads: int = 4,
                 temporal_layers: int = 1, num_pool_queries: int = 8,
                 dropout: float = 0.1, relative_query_chunk_size: int = 64,
                 time_feature_mode: str = "relative_time_delta") -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.embed_dim, self.hidden_dim, self.num_pool_queries = int(embed_dim), int(hidden_dim), int(num_pool_queries)
        if time_feature_mode not in {"none", "relative_time", "relative_time_delta"}:
            raise ValueError("time_feature_mode must be none, relative_time or relative_time_delta.")
        self.time_feature_mode = time_feature_mode
        self.token = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, hidden_dim), nn.GELU())
        self.relative_time = TimeEmbedding(hidden_dim)
        self.log_time = TimeEmbedding(hidden_dim)
        self.log_delta = TimeEmbedding(hidden_dim)
        self.layers = nn.ModuleList([RelativeTimeTransformerLayer(hidden_dim, num_heads, dropout)
                                     for _ in range(int(temporal_layers))])
        for layer in self.layers:
            layer.relative_query_chunk_size = int(relative_query_chunk_size)
        self.pool_queries = nn.Parameter(torch.empty(1, self.num_pool_queries, hidden_dim))
        nn.init.normal_(self.pool_queries, std=hidden_dim ** -0.5)
        self.pool = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.pool_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _embed_valid(module: nn.Module, value: torch.Tensor, seq_pad_mask: torch.Tensor) -> torch.Tensor:
        """Never send padding rows through a biased nonlinear time MLP."""
        out = value.new_zeros((*value.shape, module.mlp[-1].out_features))
        valid = ~seq_pad_mask
        if valid.any():
            # ``value`` remains FP32 while CUDA autocast executes the MLP in
            # BF16.  The indexed destination is FP32, so cast explicitly.
            out[valid] = module(value[valid].unsqueeze(-1)).to(dtype=out.dtype)
        return out

    def forward(self, latent_seq: torch.Tensor, time_min: torch.Tensor,
                delta_time_min: torch.Tensor, seq_pad_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _validate_probe_inputs(latent_seq, time_min, delta_time_min, seq_pad_mask)
        first = time_min.masked_fill(seq_pad_mask, float("inf")).amin(dim=1, keepdim=True)
        relative = torch.where(seq_pad_mask, torch.zeros_like(time_min), time_min - first).clamp_min(0.0)
        safe_delta = torch.where(seq_pad_mask, torch.zeros_like(delta_time_min), delta_time_min).clamp_min(0.0)
        log_time = torch.log1p(relative)
        log_delta = torch.log1p(safe_delta)
        x = self.token(latent_seq)
        if self.time_feature_mode != "none":
            x = x + self._embed_valid(self.relative_time, relative, seq_pad_mask)
        if self.time_feature_mode == "relative_time_delta":
            x = x + self._embed_valid(self.log_time, log_time, seq_pad_mask)
            x = x + self._embed_valid(self.log_delta, log_delta, seq_pad_mask)
        x = x.masked_fill(seq_pad_mask.unsqueeze(-1), 0.0)
        for layer in self.layers:
            # Passing signed_log_dt=None makes the Phase-1 layer construct it
            # only per query chunk; do not use RelativeTimeTransformerEncoder.
            x = layer(x, seq_pad_mask, time_min, signed_log_dt=None,
                      disable_relative_time_bias=self.time_feature_mode == "none")
        query = self.pool_queries.expand(x.size(0), -1, -1)
        pooled, _ = self.pool(query, x, x, key_padding_mask=seq_pad_mask, need_weights=False)
        tokens = self.pool_norm(pooled)
        return tokens, tokens.flatten(start_dim=1)


class SourceAwareProbeEncoder(nn.Module):
    """The single formal reader for frozen Phase-1 sequence representations.

    ``z_set`` has no longitudinal time information, so its one temporal layer
    is the same relative-time layer used by Phase 1.  ``z_context`` already
    contains that information and therefore uses an ordinary Transformer layer
    without reading the externally supplied timestamps.
    """

    def __init__(self, embed_dim: int, source_name: str, hidden_dim: int = 256,
                 num_heads: int = 4, temporal_layers: int = 1,
                 num_pool_queries: int = 4, dropout: float = 0.1,
                 relative_query_chunk_size: int = 64, ffn_dim: int = 1024,
                 **_: Any) -> None:
        super().__init__()
        if source_name not in CANONICAL_LATENT_SOURCES:
            raise ValueError("source_name must be z_set or z_context.")
        if temporal_layers != 1:
            raise ValueError("The formal probe uses exactly one temporal layer.")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.source_name = source_name
        self.hidden_dim = int(hidden_dim)
        self.num_pool_queries = int(num_pool_queries)
        self.token = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, hidden_dim), nn.GELU())
        if source_name == "z_set":
            self.temporal = RelativeTimeTransformerLayer(hidden_dim, num_heads, dropout)
            self.temporal.relative_query_chunk_size = int(relative_query_chunk_size)
        else:
            self.temporal = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=int(ffn_dim),
                dropout=dropout, activation="gelu", batch_first=True,
            )
        self.pool_queries = nn.Parameter(torch.empty(1, self.num_pool_queries, hidden_dim))
        nn.init.normal_(self.pool_queries, std=hidden_dim ** -0.5)
        self.pool = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.pool_norm = nn.LayerNorm(hidden_dim)

    def forward(self, latent_seq: torch.Tensor, time_min: torch.Tensor,
                delta_time_min: torch.Tensor, seq_pad_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _validate_probe_inputs(latent_seq, time_min, delta_time_min, seq_pad_mask)
        x = self.token(latent_seq).masked_fill(seq_pad_mask.unsqueeze(-1), 0.0)
        if self.source_name == "z_set":
            x = self.temporal(x, seq_pad_mask, time_min, signed_log_dt=None)
        else:
            x = self.temporal(x, src_key_padding_mask=seq_pad_mask)
            x = x.masked_fill(seq_pad_mask.unsqueeze(-1), 0.0)
        query = self.pool_queries.expand(x.size(0), -1, -1)
        # Pooling has only 4/8 learned queries but attends across the complete
        # admission.  Keep this score calculation FP32 under mixed precision;
        # it is inexpensive relative to temporal attention and avoids BF16
        # softmax overflow on rare very long admissions.
        if x.device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=False):
                tokens, _ = self.pool(query.float(), x.float(), x.float(), key_padding_mask=seq_pad_mask, need_weights=False)
        else:
            tokens, _ = self.pool(query, x, x, key_padding_mask=seq_pad_mask, need_weights=False)
        tokens = self.pool_norm(tokens)
        return tokens, tokens.flatten(start_dim=1)


class SimpleProbeEncoder(nn.Module):
    """Search-only pooling encoders sharing the canonical exact-time input contract."""

    def __init__(self, embed_dim: int, hidden_dim: int = 128, num_heads: int = 4,
                 num_pool_queries: int = 1, dropout: float = 0.1,
                 pooling: str = "query", time_feature_mode: str = "none", **_: Any) -> None:
        super().__init__()
        if pooling not in {"mean", "mean_last", "query"}:
            raise ValueError(f"Unsupported simple pooling mode {pooling!r}.")
        self.hidden_dim, self.pooling = int(hidden_dim), pooling
        self.num_pool_queries = 2 if pooling == "mean_last" else (1 if pooling == "mean" else int(num_pool_queries))
        self.token = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, hidden_dim), nn.GELU())
        if time_feature_mode not in {"none", "relative_time", "relative_time_delta"}:
            raise ValueError("Simple probe time_feature_mode must be none, relative_time or relative_time_delta.")
        self.time_feature_mode = time_feature_mode
        self.relative_time = TimeEmbedding(hidden_dim)
        self.log_delta = TimeEmbedding(hidden_dim)
        if pooling == "query":
            self.pool_queries = nn.Parameter(torch.empty(1, self.num_pool_queries, hidden_dim))
            nn.init.normal_(self.pool_queries, std=hidden_dim ** -0.5)
            self.pool = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.pool_norm = nn.LayerNorm(hidden_dim)

    def forward(self, latent_seq: torch.Tensor, time_min: torch.Tensor,
                delta_time_min: torch.Tensor, seq_pad_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _validate_probe_inputs(latent_seq, time_min, delta_time_min, seq_pad_mask)
        x = self.token(latent_seq).masked_fill(seq_pad_mask.unsqueeze(-1), 0.0)
        if self.time_feature_mode != "none":
            first = time_min.masked_fill(seq_pad_mask, float("inf")).amin(dim=1, keepdim=True)
            relative = torch.where(seq_pad_mask, torch.zeros_like(time_min), time_min - first).clamp_min(0.0)
            x = x + AdaptiveTimeAwareProbeEncoder._embed_valid(self.relative_time, relative, seq_pad_mask)
            if self.time_feature_mode == "relative_time_delta":
                safe_delta = torch.where(seq_pad_mask, torch.zeros_like(delta_time_min), delta_time_min).clamp_min(0.0)
                x = x + AdaptiveTimeAwareProbeEncoder._embed_valid(self.log_delta, torch.log1p(safe_delta), seq_pad_mask)
            x = x.masked_fill(seq_pad_mask.unsqueeze(-1), 0.0)
        valid = (~seq_pad_mask).unsqueeze(-1)
        if self.pooling == "mean":
            tokens = (x * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        elif self.pooling == "mean_last":
            mean = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
            last = x[torch.arange(x.size(0), device=x.device), (~seq_pad_mask).sum(dim=1) - 1]
            tokens = torch.stack([mean, last], dim=1)
        else:
            query = self.pool_queries.expand(x.size(0), -1, -1)
            tokens, _ = self.pool(query, x, x, key_padding_mask=seq_pad_mask, need_weights=False)
        tokens = self.pool_norm(tokens)
        return tokens, tokens.flatten(start_dim=1)


class RawLinearProbeEncoder(nn.Module):
    """Parameter-free raw latent pooling for representation diagnostics."""

    def __init__(self, embed_dim: int, pooling: str) -> None:
        super().__init__()
        if pooling not in {"mean", "mean_last"}:
            raise ValueError("Raw linear encoder supports mean or mean_last pooling only.")
        self.hidden_dim = int(embed_dim)
        self.num_pool_queries = 2 if pooling == "mean_last" else 1
        self.pooling = pooling

    def forward(self, latent_seq: torch.Tensor, time_min: torch.Tensor,
                delta_time_min: torch.Tensor, seq_pad_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _validate_probe_inputs(latent_seq, time_min, delta_time_min, seq_pad_mask)
        valid = (~seq_pad_mask).unsqueeze(-1)
        mean = (latent_seq * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        if self.pooling == "mean":
            tokens = mean.unsqueeze(1)
        else:
            last = latent_seq[torch.arange(latent_seq.size(0), device=latent_seq.device), (~seq_pad_mask).sum(dim=1) - 1]
            tokens = torch.stack([mean, last], dim=1)
        return tokens, tokens.flatten(start_dim=1)


class _ProbeBase(nn.Module):
    def __init__(self, encoder: nn.Module, dropout: float, linear_head: bool = False) -> None:
        super().__init__()
        self.encoder = encoder
        input_dim = int(encoder.hidden_dim * encoder.num_pool_queries)
        self.output_dim = input_dim if linear_head else int(encoder.hidden_dim)
        self.trunk = (nn.Identity() if linear_head else nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, encoder.hidden_dim), nn.GELU(), nn.Dropout(dropout)))

    def encode(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        _, patient = self.encoder(batch["latent_seq"], batch["time_min"], batch["delta_time_min"], batch["seq_pad_mask"])
        return self.trunk(patient)


class BinaryOutcomeProbe(_ProbeBase):
    def __init__(self, encoder: nn.Module, dropout: float = 0.1, linear_head: bool = False) -> None:
        super().__init__(encoder, dropout, linear_head)
        self.head = nn.Linear(self.output_dim, 1)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.encode(batch)).squeeze(-1)


class OrdinalLOSProbe(_ProbeBase):
    def __init__(self, encoder: nn.Module, dropout: float = 0.1, linear_head: bool = False) -> None:
        super().__init__(encoder, dropout, linear_head)
        self.head = nn.Linear(self.output_dim, 2)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.head(self.encode(batch))
        return logits[:, 0], logits[:, 1]

    @staticmethod
    def probabilities(logit_ge3: torch.Tensor, logit_conditional_ge7: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p3 = torch.sigmoid(logit_ge3)
        return p3, p3 * torch.sigmoid(logit_conditional_ge7)

    @staticmethod
    def loss(logit_ge3: torch.Tensor, logit_conditional_ge7: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ge3_loss, conditional_loss, observed_ge3 = OrdinalLOSProbe.loss_components(
            logit_ge3, logit_conditional_ge7, target
        )
        result = ge3_loss.mean()
        if observed_ge3.any():
            result = result + conditional_loss[observed_ge3].mean()
        return result

    @staticmethod
    def loss_per_patient(logit_ge3: torch.Tensor, logit_conditional_ge7: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ge3_loss, conditional_loss, _ = OrdinalLOSProbe.loss_components(logit_ge3, logit_conditional_ge7, target)
        # This is an unnormalised diagnostic per-patient quantity.  The
        # training loop owns the two-denominator LOS objective so that a
        # dynamic microbatch split cannot change the gradient.
        return ge3_loss + conditional_loss

    @staticmethod
    def loss_components(logit_ge3: torch.Tensor, logit_conditional_ge7: torch.Tensor,
                        target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if target.ndim != 2 or target.size(1) != 2:
            raise ValueError("LOS target must be [B,2]: y_ge3, y_ge7.")
        y3, y7 = target[:, 0], target[:, 1]
        if (y7 > y3).any():
            raise ValueError("LOS targets must satisfy y_ge7 <= y_ge3.")
        ge3_loss = F.binary_cross_entropy_with_logits(logit_ge3, y3, reduction="none")
        observed_ge3 = y3.bool()
        conditional_loss = torch.zeros_like(ge3_loss)
        if observed_ge3.any():
            conditional_loss[observed_ge3] = F.binary_cross_entropy_with_logits(
                logit_conditional_ge7[observed_ge3], y7[observed_ge3], reduction="none"
            )
        return ge3_loss, conditional_loss, observed_ge3


class MultiLabelProbe(_ProbeBase):
    def __init__(self, encoder: nn.Module, num_labels: int, dropout: float = 0.1, linear_head: bool = False) -> None:
        super().__init__(encoder, dropout, linear_head)
        self.head = nn.Linear(self.output_dim, int(num_labels))

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.encode(batch))


class CompetitiveDRGProbe(_ProbeBase):
    def __init__(self, encoder: nn.Module, num_labels: int, dropout: float = 0.1, linear_head: bool = False) -> None:
        super().__init__(encoder, dropout, linear_head)
        self.label_head = nn.Linear(self.output_dim, int(num_labels))
        self.cardinality_head = nn.Linear(self.output_dim, 3)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(batch)
        return self.label_head(h), self.cardinality_head(h)

    @staticmethod
    def loss(label_logits: torch.Tensor, cardinality_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return CompetitiveDRGProbe.loss_per_patient(label_logits, cardinality_logits, target).mean()

    @staticmethod
    def loss_per_patient(label_logits: torch.Tensor, cardinality_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Permutation-invariant likelihood for targets encoded as BxL multi-hot."""
        if target.ndim != 2 or target.shape != label_logits.shape:
            raise ValueError("DRG target must be a [B,L] multi-hot tensor matching label logits.")
        cardinality = target.sum(dim=1).long()
        if (cardinality > 2).any():
            raise ValueError("Competitive DRG targets may contain at most two labels.")
        loss = F.cross_entropy(cardinality_logits, cardinality, reduction="none")
        terms = torch.zeros_like(loss)
        for row in range(target.size(0)):
            labels = torch.where(target[row].bool())[0]
            if labels.numel() == 1:
                terms[row] = -F.log_softmax(label_logits[row], dim=-1)[labels[0]]
            elif labels.numel() == 2:
                a, b = labels[0], labels[1]
                logp = F.log_softmax(label_logits[row], dim=-1)
                ab = logp[a] + F.log_softmax(label_logits[row].masked_fill(torch.arange(label_logits.size(1), device=target.device) == a, float("-inf")), dim=-1)[b]
                ba = logp[b] + F.log_softmax(label_logits[row].masked_fill(torch.arange(label_logits.size(1), device=target.device) == b, float("-inf")), dim=-1)[a]
                terms[row] = -torch.logsumexp(torch.stack([ab, ba]), dim=0)
        return loss + terms

    @staticmethod
    def decode(label_logits: torch.Tensor, cardinality_logits: torch.Tensor) -> torch.Tensor:
        cardinality = cardinality_logits.argmax(dim=-1)
        out = torch.zeros_like(label_logits, dtype=torch.bool)
        for k in (1, 2):
            rows = torch.where(cardinality == k)[0]
            if rows.numel():
                ids = label_logits[rows].topk(k, dim=-1).indices
                out[rows.unsqueeze(1), ids] = True
        return out


def asymmetric_loss(logits: torch.Tensor, target: torch.Tensor, gamma_pos: float = 0.5,
                    gamma_neg: float = 4.0, clip: float = 0.05) -> torch.Tensor:
    return asymmetric_loss_per_patient(logits, target, gamma_pos, gamma_neg, clip).mean()


def asymmetric_loss_per_patient(logits: torch.Tensor, target: torch.Tensor, gamma_pos: float = 0.5,
                                gamma_neg: float = 4.0, clip: float = 0.05) -> torch.Tensor:
    # ``gamma_pos < 1`` has an infinite derivative at exactly ``p == 1``.
    # BF16 encoder logits can saturate sigmoid even when the scalar loss is
    # finite, producing NaN gradients through the focal weight.  The same
    # bounded probability is therefore used for both log and focal terms.
    prob = torch.sigmoid(logits).clamp(min=1e-6, max=1.0 - 1e-6)
    if clip > 0:
        prob_neg = (1.0 - prob + clip).clamp(max=1.0)
    else:
        prob_neg = 1.0 - prob
    loss = target * torch.log(prob) + (1.0 - target) * torch.log(prob_neg.clamp_min(1e-6))
    weight = torch.pow(1.0 - prob, gamma_pos) * target + torch.pow(prob, gamma_neg) * (1.0 - target)
    return -(weight * loss).mean(dim=1)




def consistency_loss(clean_logits: torch.Tensor, thinned_logits: torch.Tensor, task: str) -> torch.Tensor:
    if task == "drg":
        clean_label, clean_card = clean_logits
        thin_label, thin_card = thinned_logits
        return 0.5 * (F.kl_div(F.log_softmax(clean_card, -1), F.softmax(thin_card.detach(), -1), reduction="batchmean") +
                      F.kl_div(F.log_softmax(thin_card, -1), F.softmax(clean_card.detach(), -1), reduction="batchmean")) + F.mse_loss(clean_label, thin_label)
    if isinstance(clean_logits, tuple):
        return sum(F.mse_loss(a, b) for a, b in zip(clean_logits, thinned_logits))
    return F.mse_loss(clean_logits, thinned_logits)


def _vocab_hash(vocab: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(vocab), sort_keys=True, default=str).encode()).hexdigest()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rng_state() -> Dict[str, Any]:
    return {"python": random.getstate(), "torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if state.get("cuda") is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Probe resume checkpoint contains CUDA RNG state but CUDA is unavailable.")
        torch.cuda.set_rng_state_all(state["cuda"])


def _write_epoch_log(csv_path: str, jsonl_path: str, row: Mapping[str, Any]) -> None:
    serializable = {key: value for key, value in row.items() if not isinstance(value, (list, dict))}
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serializable))
        if new_file:
            writer.writeheader()
        writer.writerow(serializable)
    with open(jsonl_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), default=str) + "\n")


def _scalar_metric_fields(prefix: str, metrics: Mapping[str, Any]) -> Dict[str, float]:
    """Flatten scalar metric output for human-readable epoch logs."""
    fields: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            fields[f"{prefix}{key}"] = float(value)
    return fields


def _code_provenance(resolved_checkpoint: str, config_path: Optional[str]) -> Dict[str, Any]:
    """Record the concrete code/checkpoint inputs of a probe run."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = {"src.probes": __file__}
    try:
        import src.cache_bank as cache_bank
        import src.train as train
        files.update({"src.cache_bank": cache_bank.__file__, "src.train": train.__file__})
    except ImportError:
        pass
    hashes = {name: _sha256_file(path) for name, path in files.items() if path and os.path.isfile(path)}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "phase1_checkpoint_path": resolved_checkpoint,
        "phase1_checkpoint_sha256": _sha256_file(resolved_checkpoint),
        "config_path": os.path.realpath(config_path) if config_path else None,
        "config_sha256": _sha256_file(config_path) if config_path and os.path.isfile(config_path) else None,
        "source_files": {name: os.path.realpath(path) for name, path in files.items() if path},
        "source_sha256": hashes,
        "git_commit": commit,
        "git_dirty": dirty,
        "command": list(sys.argv),
    }


def _load_hcup_vocab(mapping_path: str) -> Dict[str, Any]:
    with open(mapping_path, encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError("HCUP mapping artifact must be a JSON list.")
    ordered = sorted((int(row["index"]), str(row["hcup_label"])) for row in rows)
    if [index for index, _ in ordered] != list(range(26)):
        raise ValueError("HCUP mapping artifact must define exactly ordered indices 0..25.")
    codes = [label for _, label in ordered]
    return {"label_col": "HCUP", "codes": codes, "code_to_idx": {code: i for i, code in enumerate(codes)},
            "include_other_bucket": False, "mapping_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()}


def encode_hcup_target(raw_value: Any) -> torch.Tensor:
    """Encode the HCUP JSON integer-index contract without label-name matching."""
    if not isinstance(raw_value, str):
        raise ValueError("HCUP values must be JSON lists encoded as strings.")
    try:
        indices = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError("HCUP value is not valid JSON.") from exc
    if not isinstance(indices, list):
        raise ValueError("HCUP value must decode to a JSON list of integer indices.")
    target = torch.zeros(26, dtype=torch.float32)
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("HCUP indices must be integers, not label strings or booleans.")
        if not 0 <= index < 26:
            raise ValueError(f"HCUP index {index} is outside the supported range [0,25].")
        target[index] = 1.0
    return target


def _targets_for_task(labels_csv: str, task: str, cfg: Mapping[str, Any],
                      vocab_override: Optional[Mapping[str, Any]] = None,
                      hcup_mapping_json: Optional[str] = None,
                      allowed_splits: Optional[Sequence[str]] = None) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    task = str(task).lower()
    def restrict(frame: pd.DataFrame) -> pd.DataFrame:
        if allowed_splits is None:
            return frame
        if "split" not in frame:
            raise RuntimeError("Formal probe target loading requires labels.csv to contain split.")
        allowed = {str(value) for value in allowed_splits}
        return frame.loc[frame["split"].astype(str).isin(allowed)].copy()
    if task in {"los_ge_3d", "los_ge_7d"}:
        frame = restrict(load_labels_csv(labels_csv))
        if "label_los_days" not in frame:
            raise ValueError("labels.csv lacks label_los_days for LOS.")
        threshold = 3 if task == "los_ge_3d" else 7
        los = pd.to_numeric(frame["label_los_days"], errors="coerce")
        valid = los.notna() & np.isfinite(los.to_numpy(dtype=np.float64))
        frame = frame.loc[valid].copy()
        frame["label_los_days"] = los.loc[valid].astype(float)
        return {str(r.hadm_id): float(r.label_los_days >= threshold) for r in frame.itertuples()}, None
    if task in {"mortality", "readmission"}:
        frame = restrict(load_labels_csv(labels_csv))
        column = TASK_TO_LABEL_COL[task]
        if column not in frame:
            raise ValueError(f"labels.csv lacks required {column!r} for {task}.")
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.notna() & np.isfinite(values.to_numpy(dtype=np.float64))
        frame = frame.loc[valid].copy()
        frame[column] = values.loc[valid].astype(float)
        return {str(r.hadm_id): float(getattr(r, column)) for r in frame.itertuples()}, None
    if task == "los":
        frame = restrict(load_labels_csv(labels_csv))
        if "label_los_days" not in frame:
            raise ValueError("labels.csv lacks label_los_days for LOS.")
        return {str(r.hadm_id): [float(r.label_los_days >= 3), float(r.label_los_days >= 7)] for r in frame.itertuples()}, None
    if task not in {"icd", "hcup", "drg"}:
        raise ValueError(f"Unknown exact-time probe task: {task}")
    task_cfg = dict(cfg.get("tasks", {}).get(task, {}))
    label_col = str(task_cfg.get("label_col", f"{task}_codes"))
    if task == "hcup":
        if vocab_override is None:
            if not hcup_mapping_json:
                raise ValueError("HCUP requires --hcup_mapping_json; no implicit label order is allowed.")
            vocab = _load_hcup_vocab(hcup_mapping_json)
        else:
            vocab = dict(vocab_override)
            if len(vocab.get("codes", [])) != 26:
                raise ValueError("Checkpoint HCUP vocabulary must contain exactly 26 mapping-ordered labels.")
        frame = restrict(load_labels_csv(labels_csv))
        if "hadm_id" not in frame or label_col not in frame:
            raise ValueError(f"labels.csv lacks required HCUP columns 'hadm_id' and {label_col!r}.")
        return {str(row.hadm_id): encode_hcup_target(getattr(row, label_col)) for row in frame.itertuples()}, vocab
    # Restrict the CSV before parsing phenotype strings.  Formal training must
    # not inspect malformed or otherwise invalid test targets.
    frame = restrict(load_labels_csv(labels_csv))
    required = {"hadm_id", "split", label_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"labels.csv missing columns: {missing}")
    frame = frame.copy()
    frame["hadm_id_str"] = frame["hadm_id"].astype(str)
    frame["parsed_phenotype_codes"] = frame[label_col].apply(parse_phenotype_code_list)
    vocab = dict(vocab_override) if vocab_override is not None else build_phenotype_vocab_metadata(
        frame, label_col, int(task_cfg.get("min_code_count", 1)), task_cfg.get("max_codes"),
        bool(task_cfg.get("include_other_bucket", False)),
    )
    targets = {str(r.hadm_id): encode_phenotype_target(
        r.parsed_phenotype_codes, vocab["code_to_idx"], bool(vocab["include_other_bucket"])
    ) for r in restrict(frame).itertuples()}
    return targets, vocab


def _fit_multilabel_thresholds(target: torch.Tensor, logits: torch.Tensor, vocab: Mapping[str, Any]) -> Dict[str, Any]:
    grid = np.arange(0.05, 1.0, 0.05)
    y, p = target.float().numpy().astype(bool), torch.sigmoid(logits.float()).numpy()
    global_f1 = [2 * ((p >= t) & y).sum() / max(2 * ((p >= t) & y).sum() + ((p >= t) & ~y).sum() + (~(p >= t) & y).sum(), 1) for t in grid]
    global_threshold = float(grid[int(np.argmax(global_f1))])
    thresholds: List[float] = []
    fallback_label_ids: List[int] = []
    for j in range(y.shape[1]):
        if not y[:, j].any():
            thresholds.append(global_threshold)
            fallback_label_ids.append(j)
            continue
        f1 = [2 * ((p[:, j] >= t) & y[:, j]).sum() / max(2 * ((p[:, j] >= t) & y[:, j]).sum() + ((p[:, j] >= t) & ~y[:, j]).sum() + (~(p[:, j] >= t) & y[:, j]).sum(), 1) for t in grid]
        thresholds.append(float(grid[int(np.argmax(f1))]))
    if not np.isfinite(thresholds).all() or not np.all((np.asarray(thresholds) > 0) & (np.asarray(thresholds) < 1)):
        raise RuntimeError("Validation threshold fitting produced invalid thresholds.")
    return {"vocab_hash": _vocab_hash(vocab), "per_label": thresholds, "grid": grid.tolist(),
            "fitting_split": "validation", "global_fallback_threshold": global_threshold,
            "fallback_label_ids": fallback_label_ids}


def multilabel_threshold_metrics(target: torch.Tensor, logits: torch.Tensor, thresholds: Mapping[str, Any]) -> Dict[str, float]:
    values = np.asarray(thresholds.get("per_label", []), dtype=np.float64)
    if values.shape != (target.size(1),) or not np.isfinite(values).all() or not np.all((values > 0) & (values < 1)):
        raise RuntimeError("Invalid multilabel threshold artifact.")
    truth = target.float().numpy().astype(bool)
    pred = torch.sigmoid(logits.float()).numpy() >= values[None, :]
    tp, fp, fn = (pred & truth).sum(), (pred & ~truth).sum(), (~pred & truth).sum()
    per_f1 = []
    for label in range(truth.shape[1]):
        ltp, lfp, lfn = (pred[:, label] & truth[:, label]).sum(), (pred[:, label] & ~truth[:, label]).sum(), (~pred[:, label] & truth[:, label]).sum()
        per_f1.append(float(2 * ltp / max(2 * ltp + lfp + lfn, 1)))
    return {"threshold_micro_f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
            "threshold_macro_f1": float(np.mean(per_f1)), "threshold_precision": float(tp / max(tp + fp, 1)),
            "threshold_recall": float(tp / max(tp + fn, 1)), "mean_predicted_label_count": float(pred.sum(axis=1).mean())}


def _make_task_model(task: str, embed_dim: int, cfg: Mapping[str, Any], num_labels: Optional[int],
                     source_name: str = "z_context") -> nn.Module:
    encoder_cfg = dict(cfg["encoder"])
    variant = str(encoder_cfg.pop("architecture_variant", "source_aware"))
    linear_head = variant in {"linear_mean", "linear_mean_last", "raw_mean_linear", "raw_mean_last_linear"}
    if variant == "source_aware":
        if source_name not in CANONICAL_LATENT_SOURCES:
            raise ValueError("Formal probe source must be z_set or z_context.")
        encoder_cfg["num_pool_queries"] = 4 if task in {"mortality", "los_ge_3d", "los_ge_7d", "readmission"} else 8
        encoder = SourceAwareProbeEncoder(embed_dim=embed_dim, source_name=source_name, **encoder_cfg)
    elif variant == "raw_mean_linear":
        encoder = RawLinearProbeEncoder(embed_dim, "mean")
    elif variant == "raw_mean_last_linear":
        encoder = RawLinearProbeEncoder(embed_dim, "mean_last")
    elif variant == "linear_mean":
        encoder = SimpleProbeEncoder(embed_dim=embed_dim, pooling="mean", **encoder_cfg)
    elif variant == "linear_mean_last":
        encoder = SimpleProbeEncoder(embed_dim=embed_dim, pooling="mean_last", **encoder_cfg)
    elif variant == "mlp_mean_last":
        encoder = SimpleProbeEncoder(embed_dim=embed_dim, pooling="mean_last", **encoder_cfg)
    elif variant == "single_query_pool":
        encoder_cfg["num_pool_queries"] = 1
        encoder = SimpleProbeEncoder(embed_dim=embed_dim, pooling="query", **encoder_cfg)
    elif variant == "multi_query_pool":
        encoder = SimpleProbeEncoder(embed_dim=embed_dim, pooling="query", **encoder_cfg)
    elif variant == "time_transformer_single_query":
        encoder_cfg["num_pool_queries"] = 1
        encoder = AdaptiveTimeAwareProbeEncoder(embed_dim=embed_dim, **encoder_cfg)
    elif variant == "time_transformer_multi_query":
        encoder = AdaptiveTimeAwareProbeEncoder(embed_dim=embed_dim, **encoder_cfg)
    else:
        raise ValueError(f"Unknown probe architecture_variant={variant!r}.")
    dropout = float(encoder_cfg.get("dropout", 0.1))
    if task in {"mortality", "los_ge_3d", "los_ge_7d", "readmission"}:
        return BinaryOutcomeProbe(encoder, dropout, linear_head)
    if task == "los":
        return OrdinalLOSProbe(encoder, dropout, linear_head)
    if task == "drg":
        return CompetitiveDRGProbe(encoder, int(num_labels), dropout, linear_head)
    return MultiLabelProbe(encoder, int(num_labels), dropout, linear_head)


def build_exact_time_probe(*, task: str, embed_dim: int, cfg: Mapping[str, Any],
                           num_labels: Optional[int] = None,
                           source_name: str = "z_context") -> nn.Module:
    """Public, canonical probe factory used by orchestration code.

    The implementation deliberately delegates to the existing private factory;
    this keeps task heads, losses and architecture variants single-sourced.
    """
    return _make_task_model(task, embed_dim, cfg, num_labels, source_name)


def _validate_task_label_dimension(task: str, vocab: Optional[Mapping[str, Any]], task_cfg: Mapping[str, Any]) -> None:
    expected = {"icd": 2048, "hcup": 26, "drg": 512}
    if task not in expected:
        return
    actual = len(vocab.get("codes", [])) if vocab is not None else 0
    if actual != expected[task]:
        raise RuntimeError(
            f"{task.upper()} exact-time task requires {expected[task]} labels, but vocabulary construction produced {actual}. "
            f"min_code_count={task_cfg.get('min_code_count')}, max_codes={task_cfg.get('max_codes')}; refusing a silent smaller task."
        )


def resolve_runtime_precision(device: torch.device, precision: str) -> str:
    if device.type != "cuda" or str(precision) in {"32", "32-true", "fp32"}:
        return "fp32"
    if str(precision) == "bf16-mixed":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if str(precision) == "fp16-mixed":
        return "fp16"
    raise ValueError(f"Unsupported exact-time probe precision {precision!r}.")


def _autocast_context(device: torch.device, precision: str):
    runtime = resolve_runtime_precision(device, precision)
    if runtime == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if runtime == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _make_grad_scaler(device: torch.device, runtime_precision: str) -> Optional[Any]:
    """Create a scaler only for CUDA FP16, across supported PyTorch APIs."""
    if device.type != "cuda" or runtime_precision != "fp16":
        return None
    scaler_cls = getattr(getattr(torch, "amp", None), "GradScaler", None)
    if scaler_cls is not None:
        return scaler_cls("cuda", enabled=True)
    # PyTorch releases before torch.amp.GradScaler expose the CUDA namespace.
    return torch.cuda.amp.GradScaler(enabled=True)


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def _make_probe_optimizer(parameters: Any, *, device: torch.device, lr: float, weight_decay: float) -> Tuple[torch.optim.Optimizer, bool]:
    # Some deployed PyTorch/CUDA builds accept ``fused=True`` but fail only at
    # the first BF16 optimizer step when parameter/gradient layouts differ.
    # Foreach AdamW preserves the same AdamW update while batching CUDA tensor
    # operations without that fused-kernel dtype constraint.
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, foreach=True), False
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay), False


def _autotune_probe_runtime(*, model: nn.Module, dataset: Dataset, lengths: Sequence[int],
                            shard_ids: Sequence[int], task: str, cfg: Mapping[str, Any],
                            train_cfg: Mapping[str, Any], device: torch.device, seed: int,
                            source: str, output_dir: str, initial_num_workers: int) -> Dict[str, Any]:
    """Benchmark real cache batches before training and select a safe budget.

    This is intentionally a small preflight: it exercises the actual probe,
    collate path, H2D transfer, backward and optimizer step without consuming
    an epoch or changing the scientific training protocol.
    """
    if device.type != "cuda":
        result = {"enabled": False, "reason": "cuda_unavailable"}
        with open(os.path.join(output_dir, "runtime_autotune.json"), "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        return result
    candidates = [
        (elements, patients, chunk)
        for elements in (262144, 524288, 1048576, 2097152)
        for patients in (64, 128, 256)
        for chunk in ((64, 96, 128) if source == "z_set" else (int(train_cfg.get("relative_query_chunk_size", 64)),))
    ]
    original_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    rows: List[Dict[str, Any]] = []
    for elements, patients, chunk in candidates:
        model.load_state_dict(original_state, strict=True)
        torch.cuda.reset_peak_memory_stats(device)
        for layer in model.modules():
            if hasattr(layer, "relative_query_chunk_size"):
                layer.relative_query_chunk_size = int(chunk)
        sampler = ShardAwareAttentionCostBatchSampler(
            lengths, shard_ids, elements, patients, False, seed,
        )
        batches = list(sampler)
        if not batches:
            continue
        batch_indices = max(batches, key=lambda batch: max(lengths[i] for i in batch))
        loader = DataLoader(dataset, batch_sampler=[batch_indices], collate_fn=partial(canonical_probe_collate, validate=False),
                            num_workers=0, pin_memory=True)
        optimizer, _ = _make_probe_optimizer(model.parameters(), device=device,
                                             lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
        try:
            raw = next(iter(loader))
            start_h2d = torch.cuda.Event(enable_timing=True); end_h2d = torch.cuda.Event(enable_timing=True)
            start_compute = torch.cuda.Event(enable_timing=True); end_compute = torch.cuda.Event(enable_timing=True)
            start_h2d.record(); batch = _batch_to_device(raw, device); end_h2d.record()
            torch.cuda.synchronize()
            with _autocast_context(device, str(train_cfg.get("precision", "bf16-mixed"))):
                start_compute.record()
                output = model(batch)
                target = batch["target"].float()
                loss = _per_patient_task_loss(task, output, target, None,
                                              str(cfg["tasks"].get(task, {}).get("binary_loss", {}).get("type", "unweighted_bce"))).mean()
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("gradient_clip_val", 1.0)))
                optimizer.step(); end_compute.record()
            torch.cuda.synchronize()
            h2d_ms = float(start_h2d.elapsed_time(end_h2d)); compute_ms = float(start_compute.elapsed_time(end_compute))
            rows.append({"max_pairwise_elements_per_batch": elements, "max_patients_per_batch": patients,
                         "relative_query_chunk_size": chunk, "patients": len(batch_indices),
                         "h2d_ms": h2d_ms, "compute_ms": compute_ms,
                         "patients_per_second": len(batch_indices) / max((h2d_ms + compute_ms) / 1000.0, 1e-9),
                         "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)), "oom": False})
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            if "out of memory" not in str(exc).lower() and not isinstance(exc, torch.cuda.OutOfMemoryError):
                raise
            rows.append({"max_pairwise_elements_per_batch": elements, "max_patients_per_batch": patients,
                         "relative_query_chunk_size": chunk, "oom": True})
            torch.cuda.empty_cache()
        finally:
            del optimizer
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    valid = [row for row in rows if not row.get("oom")]
    if not valid:
        raise RuntimeError("Probe runtime autotune found no viable CUDA batch budget.")
    safe = [row for row in valid if int(row.get("peak_memory_bytes", 0)) <= int(total_memory * 0.85)]
    if not safe:
        raise RuntimeError("Probe runtime autotune found no CUDA budget with 15% memory headroom.")
    selected = max(safe, key=lambda row: row["patients_per_second"])
    model.load_state_dict(original_state, strict=True)
    loader_rows = []
    loader_batch = max(batches, key=lambda batch: max(lengths[i] for i in batch))
    for workers in (2, 4, 8):
        for prefetch in (2, 4):
            start = time.perf_counter()
            loader = DataLoader(dataset, batch_sampler=[loader_batch],
                                collate_fn=partial(canonical_probe_collate, validate=False),
                                num_workers=workers, pin_memory=True, persistent_workers=True,
                                prefetch_factor=prefetch)
            next(iter(loader))
            loader_rows.append({"num_workers": workers, "prefetch_factor": prefetch,
                                "first_batch_seconds": time.perf_counter() - start})
            del loader
    selected_loader = min(loader_rows, key=lambda row: row["first_batch_seconds"])
    result = {"enabled": True, "gpu_name": torch.cuda.get_device_name(device),
              "total_memory_bytes": total_memory, "memory_fraction_limit": 0.85,
              "candidates": rows, "selected": selected,
              "loader_candidates": loader_rows, "selected_num_workers": selected_loader["num_workers"],
              "selected_prefetch_factor": selected_loader["prefetch_factor"],
              "initial_num_workers": initial_num_workers,
              "shard_cache_size_candidates": [2, 4]}
    with open(os.path.join(output_dir, "runtime_autotune.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def _assert_finite(value: Any, label: str) -> None:
    tensors = value if isinstance(value, tuple) else (value,)
    if any(not torch.isfinite(t).all() for t in tensors):
        raise RuntimeError(f"Non-finite downstream probe {label}.")


def _per_patient_consistency(reference: Any, student: Any) -> torch.Tensor:
    ref = reference if isinstance(reference, tuple) else (reference,)
    out = student if isinstance(student, tuple) else (student,)
    return sum((current - target.detach()).reshape(current.size(0), -1).square().mean(dim=1)
               for target, current in zip(ref, out))


def _per_patient_task_loss(task: str, output: Any, target: torch.Tensor,
                           pos_weight: Optional[torch.Tensor], binary_loss_type: str = "unweighted_bce") -> Tuple[torch.Tensor, Any]:
    if task in {"mortality", "los_ge_3d", "los_ge_7d", "readmission"}:
        logits = output.float()
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
        if binary_loss_type == "focal":
            spec = FOCAL_LOSS_PARAMETERS.get(task)
            if spec is None:
                raise ValueError(f"No fixed focal-loss contract for task={task!r}.")
            pt = torch.where(target.bool(), torch.sigmoid(logits), torch.sigmoid(-logits))
            alpha = torch.where(target.bool(), torch.full_like(target, float(spec["alpha"])),
                                torch.full_like(target, 1.0 - float(spec["alpha"])))
            return alpha * (1.0 - pt).pow(float(spec["gamma"])) * bce, output
        return bce, output
    if task == "los":
        ge3, conditional_ge7 = output
        return OrdinalLOSProbe.loss_per_patient(ge3, conditional_ge7, target), torch.stack([ge3, conditional_ge7], dim=1)
    if task == "drg":
        return CompetitiveDRGProbe.loss_per_patient(output[0].float(), output[1].float(), target.float()), output
    # ASL includes sigmoid/log/power operations over 26--2048 labels.  Keep
    # this small loss path FP32 even when the encoder forward is BF16.
    return asymmetric_loss_per_patient(output.float(), target.float()), output


def _epoch(model: nn.Module, loader: DataLoader, task: str, device: torch.device,
           optimizer: Optional[torch.optim.Optimizer], density_cfg: Mapping[str, Any],
           collect_predictions: bool = True, pos_weight: Optional[torch.Tensor] = None,
           gradient_clip_val: Optional[float] = None, precision: str = "32-true",
           strict_finite: bool = False, progress_desc: Optional[str] = None,
           effective_batch_patients: int = 1, telemetry: Optional[Dict[str, Any]] = None,
           scaler: Optional[Any] = None) -> Tuple[float, List[str], torch.Tensor, Any]:
    train = optimizer is not None
    model.train(train)
    if not train and precision == "32-true":
        model.float()
    loss_sum = torch.zeros((), device=device); patient_count = 0; batch_count = 0
    accumulated_patients = 0; cycle_patients: List[int] = []; microbatch_sizes: List[int] = []; optimizer_steps = 0
    cycle_ge3_terms: List[torch.Tensor] = []; cycle_conditional_terms: List[torch.Tensor] = []; cycle_consistency_terms: List[torch.Tensor] = []; cycle_ge3_count = 0
    los_ge3_sum = torch.zeros((), device=device); los_conditional_sum = torch.zeros((), device=device); los_ge3_count = 0
    cycle_ge3_counts: List[int] = []
    supervised_sum = torch.zeros((), device=device); consistency_sum = torch.zeros((), device=device); last_grad_norm = None
    if train:
        optimizer.zero_grad(set_to_none=True)
    ids, targets, logits = [], [], []
    if progress_desc:
        print(f"[{progress_desc}] loader ready: {len(loader)} batches; waiting for first batch...", flush=True)
    iterator = (tqdm(loader, desc=progress_desc, unit="batch", dynamic_ncols=True, leave=False,
                     disable=not sys.stderr.isatty()) if progress_desc else loader)
    heartbeat_start = time.perf_counter()
    last_heartbeat = heartbeat_start
    finite_interval = int(density_cfg.get("finite_check_interval", 1 if strict_finite else 0))
    finite_checks = 0
    runtime_precision = resolve_runtime_precision(device, precision)
    for raw_batch in iterator:
        batch = _batch_to_device(raw_batch, device)
        if runtime_precision == "fp32" and isinstance(batch.get("latent_seq"), torch.Tensor):
            batch["latent_seq"] = batch["latent_seq"].float()
        grad_context = torch.enable_grad() if train else torch.inference_mode()
        with grad_context:
            with _autocast_context(device, precision):
                output = model(batch)
                target = batch["target"].float()
            if task == "los":
                ge3_each, conditional_each, ge3_mask = OrdinalLOSProbe.loss_components(output[0], output[1], target)
                supervised_per_patient, logit = ge3_each + conditional_each, torch.stack(output, dim=1)
            else:
                supervised_per_patient, logit = _per_patient_task_loss(
                    task, output, target, pos_weight, str(density_cfg.get("binary_loss_type", "unweighted_bce")))
            consistency_per_patient = torch.zeros_like(supervised_per_patient)
            effective_consistency_weight = float(density_cfg.get("effective_consistency_weight", 0.0))
            if train and bool(density_cfg.get("enabled", False)) and effective_consistency_weight > 0:
                # Compare clean and thinned views with dropout disabled, while
                # retaining gradients only through the thinned student branch.
                model.eval()
                with torch.no_grad():
                    reference = model(batch)
                thin = thin_probe_batch(batch, density_cfg["retention_choices"], float(density_cfg.get("protected_change_fraction", 0.10)))
                thinned = model(thin)
                model.train()
                consistency_per_patient = _per_patient_consistency(reference, thinned)
                if strict_finite or (finite_interval and batch_count % finite_interval == 0):
                    _assert_finite(thinned, "thinned logits")
                    _assert_finite(consistency_per_patient, "consistency loss")
            loss_per_patient = supervised_per_patient + effective_consistency_weight * consistency_per_patient
            loss_sum_microbatch = loss_per_patient.sum()
        if strict_finite or (finite_interval and batch_count % finite_interval == 0):
            _assert_finite(output, "clean logits")
            _assert_finite(loss_per_patient, "per-patient loss")
            finite_checks += 1
        if train:
            accumulated_patients += int(target.size(0))
            if task == "los":
                cycle_ge3_terms.append(ge3_each.sum())
                cycle_conditional_terms.append(conditional_each.sum())
                cycle_consistency_terms.append(effective_consistency_weight * consistency_per_patient.sum())
                cycle_ge3_count += int(ge3_mask.sum().detach().cpu())
            else:
                (scaler.scale(loss_sum_microbatch) if scaler is not None else loss_sum_microbatch).backward()
            if accumulated_patients >= int(effective_batch_patients):
                if task == "los":
                    cycle_loss = torch.stack(cycle_ge3_terms).sum() / accumulated_patients
                    if cycle_ge3_count:
                        cycle_loss = cycle_loss + torch.stack(cycle_conditional_terms).sum() / cycle_ge3_count
                    cycle_loss = cycle_loss + torch.stack(cycle_consistency_terms).sum() / accumulated_patients
                    (scaler.scale(cycle_loss) if scaler is not None else cycle_loss).backward()
                if scaler is not None:
                    scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None and task != "los":
                        parameter.grad.div_(accumulated_patients)
                if strict_finite:
                    for parameter in model.parameters():
                        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                            raise RuntimeError(f"Non-finite downstream probe gradient; IDs={batch['hadm_id']}, lengths={batch['lengths'].tolist()}.")
                if gradient_clip_val is not None:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_val))
                    if not torch.isfinite(norm):
                        raise RuntimeError("Non-finite downstream probe gradient norm.")
                else:
                    norm = torch.linalg.vector_norm(torch.stack([
                        parameter.grad.detach().float().norm()
                        for parameter in model.parameters() if parameter.grad is not None
                    ])) if any(parameter.grad is not None for parameter in model.parameters()) else torch.zeros((), device=device)
                if not torch.isfinite(norm):
                    raise RuntimeError("Non-finite downstream probe gradient norm.")
                last_grad_norm = float(norm.detach().float().item())
                if scaler is not None:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                if strict_finite and any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                    raise RuntimeError("Non-finite downstream probe parameter after optimizer step.")
                optimizer.zero_grad(set_to_none=True); optimizer_steps += 1
                cycle_patients.append(accumulated_patients); cycle_ge3_counts.append(cycle_ge3_count)
                accumulated_patients = 0; cycle_ge3_terms = []; cycle_conditional_terms = []; cycle_consistency_terms = []; cycle_ge3_count = 0
        loss_sum = loss_sum + loss_per_patient.detach().sum()
        if task == "los":
            los_ge3_sum = los_ge3_sum + ge3_each.detach().sum()
            los_conditional_sum = los_conditional_sum + conditional_each.detach().sum()
            los_ge3_count += int(ge3_mask.sum().detach().cpu())
        supervised_sum = supervised_sum + supervised_per_patient.detach().sum()
        consistency_sum = consistency_sum + consistency_per_patient.detach().sum()
        patient_count += int(target.size(0))
        microbatch_sizes.append(int(target.size(0)))
        batch_count += 1
        if progress_desc and batch_count % 100 == 0:
            iterator.set_postfix(loss=f"{(loss_sum / patient_count).float().item():.4f}", patients=patient_count)
        if progress_desc and (batch_count == 1 or time.perf_counter() - last_heartbeat >= 60.0):
            elapsed = time.perf_counter() - heartbeat_start
            print(
                f"[{progress_desc}] batch={batch_count}/{len(loader)} patients={patient_count} "
                f"mean_loss={(loss_sum / max(patient_count, 1)).float().item():.6f} elapsed_sec={elapsed:.1f}",
                flush=True,
            )
            last_heartbeat = time.perf_counter()
        if collect_predictions:
            ids.extend(batch["hadm_id"]); targets.append(target.detach().cpu())
            if task == "drg":
                logits.append((logit[0].detach().float().cpu(), logit[1].detach().float().cpu()))
            else:
                logits.append(logit.detach().float().cpu())
    if train and accumulated_patients:
        if task == "los":
            cycle_loss = torch.stack(cycle_ge3_terms).sum() / accumulated_patients
            if cycle_ge3_count:
                cycle_loss = cycle_loss + torch.stack(cycle_conditional_terms).sum() / cycle_ge3_count
            cycle_loss = cycle_loss + torch.stack(cycle_consistency_terms).sum() / accumulated_patients
            (scaler.scale(cycle_loss) if scaler is not None else cycle_loss).backward()
        if scaler is not None:
            scaler.unscale_(optimizer)
        for parameter in model.parameters():
            if parameter.grad is not None:
                if task != "los":
                    parameter.grad.div_(accumulated_patients)
        if gradient_clip_val is not None:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_val))
        else:
            norm = torch.linalg.vector_norm(torch.stack([
                parameter.grad.detach().float().norm()
                for parameter in model.parameters() if parameter.grad is not None
            ])) if any(parameter.grad is not None for parameter in model.parameters()) else torch.zeros((), device=device)
        if not torch.isfinite(norm):
            raise RuntimeError("Non-finite downstream probe gradient norm.")
        last_grad_norm = float(norm.detach().float().item())
        if scaler is not None:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        if strict_finite:
            _assert_finite(tuple(parameter for parameter in model.parameters()), "probe parameters after optimizer step")
        optimizer.zero_grad(set_to_none=True); optimizer_steps += 1
        cycle_patients.append(accumulated_patients); cycle_ge3_counts.append(cycle_ge3_count)
    if telemetry is not None:
        telemetry.update({"patients": patient_count, "optimizer_steps": optimizer_steps,
                          "microbatch_sizes": microbatch_sizes, "accumulation_cycle_patients": cycle_patients,
                          "accumulation_cycle_ge3_patients": cycle_ge3_counts,
                          "gradient_norm": last_grad_norm,
                          "task_loss": float((supervised_sum / max(patient_count, 1)).float().item()),
                          "consistency_loss": float((consistency_sum / max(patient_count, 1)).float().item()),
                          "finite_check_mode": "strict" if strict_finite else "epoch",
                          "finite_checks": finite_checks})
    epoch_loss = ((los_ge3_sum / max(patient_count, 1)) + (los_conditional_sum / max(los_ge3_count, 1))
                  if task == "los" else loss_sum / max(patient_count, 1))
    if not strict_finite:
        _assert_finite(epoch_loss, "epoch loss")
    if not collect_predictions:
        return (float(epoch_loss.float().item()) if patient_count else float("nan"), [], torch.empty(0), None)
    output_logits: Any = (torch.cat([x[0] for x in logits], dim=0), torch.cat([x[1] for x in logits], dim=0)) if task == "drg" else torch.cat(logits, dim=0)
    return (float(epoch_loss.float().item()) if patient_count else float("nan"), ids,
            torch.cat(targets, dim=0), output_logits)


def _drg_metrics(target: torch.Tensor, label_logits: torch.Tensor, cardinality_logits: torch.Tensor) -> Dict[str, float]:
    y = target.float().numpy().astype(bool)
    label_logits, cardinality_logits = label_logits.float(), cardinality_logits.float()
    pred = CompetitiveDRGProbe.decode(label_logits, cardinality_logits).numpy()
    tp, fp, fn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()
    micro = float(2 * tp / max(2 * tp + fp + fn, 1))
    per_label = []
    for j in range(y.shape[1]):
        ltp, lfp, lfn = (pred[:, j] & y[:, j]).sum(), (pred[:, j] & ~y[:, j]).sum(), (~pred[:, j] & y[:, j]).sum()
        if 2 * ltp + lfp + lfn:
            per_label.append(2 * ltp / (2 * ltp + lfp + lfn))
    true_k, pred_k = y.sum(axis=1), pred.sum(axis=1)
    eligible = true_k > 0
    eligible_count = int(eligible.sum())
    ranks = label_logits.argsort(dim=1, descending=True).numpy()
    if eligible_count:
        top1 = np.array([bool(y[i, ranks[i, 0]]) for i in np.where(eligible)[0]])
        top2 = np.array([y[i, ranks[i, :2]].any() for i in np.where(eligible)[0]])
        topk_recall = np.array([y[i, ranks[i, : min(5, y.shape[1])]].sum() / true_k[i] for i in np.where(eligible)[0]])
        top1_value, top2_value, top5_value = float(top1.mean()), float(top2.mean()), float(topk_recall.mean())
    else:
        top1_value = top2_value = top5_value = float("nan")
    return {"micro_f1": micro, "macro_f1": float(np.mean(per_label)) if per_label else float("nan"),
            "exact_set_match": float(np.all(pred == y, axis=1).mean()), "cardinality_accuracy": float((pred_k == true_k).mean()),
            "top1_hit": top1_value, "top2_hit": top2_value, "top5_recall": top5_value,
            "topk_eligible_patient_count": eligible_count, "supported_label_count": len(per_label)}


def _task_metric(task: str, target: torch.Tensor, logits: Any) -> Tuple[float, Dict[str, float]]:
    y = target.float().numpy()
    if task in {"mortality", "los_ge_3d", "los_ge_7d", "readmission"}:
        z = logits.float().numpy(); metrics = compute_binary_metrics(y, 1.0 / (1.0 + np.exp(-z)))
        return float(metrics.get("auprc", float("nan"))), metrics
    if task == "los":
        z = logits.float().numpy(); p3 = 1.0 / (1.0 + np.exp(-z[:, 0])); p7 = p3 * (1.0 / (1.0 + np.exp(-z[:, 1])))
        m3, m7 = compute_binary_metrics(y[:, 0], p3), compute_binary_metrics(y[:, 1], p7)
        metrics = {f"los_ge3_{k}": v for k, v in m3.items()} | {f"los_ge7_{k}": v for k, v in m7.items()}
        return 0.5 * (float(m3.get("auprc", np.nan)) + float(m7.get("auprc", np.nan))), metrics
    if task == "drg":
        metrics = _drg_metrics(target, logits[0], logits[1])
        return float(metrics["micro_f1"]), metrics
    z = logits.float().numpy(); metrics = compute_multilabel_metrics(y, 1.0 / (1.0 + np.exp(-z)), [5, 10])
    return float(metrics.get("macro_auprc" if task == "hcup" else "micro_auprc", np.nan)), metrics


def _probabilities(task: str, logits: Any) -> Any:
    if task in {"mortality", "los_ge_3d", "los_ge_7d", "readmission"}:
        return torch.sigmoid(logits.float())
    if task == "los":
        p3, p7 = OrdinalLOSProbe.probabilities(logits[0].float(), logits[1].float())
        return torch.stack([p3, p7], dim=1)
    if task == "drg":
        return torch.sigmoid(logits[0].float()), torch.softmax(logits[1].float(), dim=1)
    return torch.sigmoid(logits.float())


def _write_prediction_npz(path: str, task: str, ids: Sequence[str], target: torch.Tensor, logits: Any) -> None:
    """Optional compact artifact; probabilities are always derived from logits."""
    payload: Dict[str, Any] = {
        "hadm_id": np.asarray(list(map(str, ids))),
        "target": target.detach().cpu().numpy(),
    }
    if task == "drg":
        payload["label_logits"] = logits[0].detach().cpu().to(torch.float16).numpy()
        payload["cardinality_logits"] = logits[1].detach().cpu().to(torch.float16).numpy()
    else:
        payload["logits"] = logits.detach().cpu().to(torch.float16).numpy()
    np.savez_compressed(path, **payload)


def _smoke_indices(ids: Sequence[str], targets: Mapping[str, Any], task: str, cap: int,
                   seed: int, split: str) -> Tuple[List[int], Dict[str, int]]:
    """Deterministically sample a diagnostic cohort while preserving required labels."""
    if cap <= 0:
        raise ValueError("Smoke cohort size must be positive.")
    rng = np.random.default_rng(seed)
    all_indices = list(range(len(ids)))
    if cap >= len(all_indices):
        selected = all_indices
    else:
        groups: Dict[str, List[int]] = {}
        if task in {"mortality", "los_ge_3d", "los_ge_7d", "readmission"}:
            groups = {"negative": [i for i, hid in enumerate(ids) if float(targets[hid]) == 0.0],
                      "positive": [i for i, hid in enumerate(ids) if float(targets[hid]) == 1.0]}
            if cap < 2 or not groups["negative"] or not groups["positive"]:
                raise RuntimeError(f"{split} {task} smoke requires at least two examples and both outcome classes.")
            required = [int(rng.choice(groups["negative"])), int(rng.choice(groups["positive"]))]
        elif task == "los":
            groups = {
                "lt3": [i for i, hid in enumerate(ids) if targets[hid][0] == 0.0],
                "ge3_lt7": [i for i, hid in enumerate(ids) if targets[hid][0] == 1.0 and targets[hid][1] == 0.0],
                "ge7": [i for i, hid in enumerate(ids) if targets[hid][1] == 1.0],
            }
            required_names = ["lt3", "ge3_lt7", "ge7"] if split in {"train", "val"} else [name for name, values in groups.items() if values]
            if cap < len(required_names) or any(not groups[name] for name in required_names):
                raise RuntimeError(f"{split} LOS smoke requires the available required LOS strata and enough requested patients.")
            required = [int(rng.choice(groups[name])) for name in required_names]
        elif task == "drg":
            groups = {str(k): [] for k in range(3)}
            for i, hid in enumerate(ids):
                cardinality = int(torch.as_tensor(targets[hid]).sum().item())
                if cardinality not in {0, 1, 2}:
                    raise RuntimeError(f"DRG target for {hid} has unsupported cardinality {cardinality}.")
                groups[str(cardinality)].append(i)
            required_names = ["1"] + [name for name in ("0", "2") if groups[name]]
            if cap < len(required_names) or not groups["1"]:
                raise RuntimeError(f"{split} DRG smoke requires cardinality-1 and each present cardinality stratum.")
            required = [int(rng.choice(groups[name])) for name in required_names]
        else:
            positive = [i for i, hid in enumerate(ids) if bool(torch.as_tensor(targets[hid]).sum().item())]
            if not positive:
                raise RuntimeError(f"{split} {task} smoke requires at least one positive label.")
            required = [int(rng.choice(positive))]
            groups = {"positive_admissions": positive}
        remaining = [i for i in all_indices if i not in set(required)]
        take = cap - len(required)
        selected = sorted(required + rng.choice(remaining, size=take, replace=False).tolist())
    distribution: Dict[str, int] = {"selected_patients": len(selected)}
    if task in {"mortality", "los_ge_3d", "los_ge_7d", "readmission"}:
        distribution.update({"negative": sum(float(targets[ids[i]]) == 0.0 for i in selected),
                             "positive": sum(float(targets[ids[i]]) == 1.0 for i in selected)})
    elif task == "los":
        distribution.update({"lt3": sum(targets[ids[i]][0] == 0.0 for i in selected),
                             "ge3_lt7": sum(targets[ids[i]][0] == 1.0 and targets[ids[i]][1] == 0.0 for i in selected),
                             "ge7": sum(targets[ids[i]][1] == 1.0 for i in selected)})
    elif task == "drg":
        distribution.update({f"cardinality_{k}": sum(int(torch.as_tensor(targets[ids[i]]).sum()) == k for i in selected) for k in range(3)})
    else:
        distribution["positive_labels"] = int(sum(torch.as_tensor(targets[ids[i]]).sum().item() for i in selected))
    return selected, distribution


def run_probe_phase(*, phase: str, args: Any, output_dir: str, cache_root: str, cfg_dict: Dict[str, Any],
                    evaluate_test: bool = False) -> Dict[str, Any]:
    """Train exactly one canonical task without any legacy source mapping."""
    run_start = time.perf_counter()
    if phase not in {"prediction_probe", "phenotype_probe", "probe_train"}:
        raise ValueError("Only canonical exact-time probe training phases are supported.")
    cfg = dict(cfg_dict.get("probe", {}))
    if cfg.get("architecture_version") != PROBE_ARCHITECTURE_VERSION:
        raise ValueError("configs.probe.architecture_version is missing or incompatible.")
    task = str(args.prediction_task if phase in {"prediction_probe", "probe_train"} else args.task_type).lower()
    source = str(cfg.get("source_name", "z_context"))
    if source not in CANONICAL_LATENT_SOURCES:
        raise ValueError("probe.source_name must be z_context or z_set.")
    resolved_ckpt = os.path.realpath(str(getattr(args, "phase1_ckpt", "")))
    if not resolved_ckpt or not os.path.isfile(resolved_ckpt):
        raise ValueError("Exact-time probe training requires an existing --phase1_ckpt.")
    checkpoint_sha256 = _sha256_file(resolved_ckpt)
    provenance = _code_provenance(resolved_ckpt, getattr(args, "config", None))
    manifest_path = os.path.join(cache_root, "cache_manifest.json")
    if not os.path.isfile(manifest_path):
        raise RuntimeError("Exact-time probe training requires a checkpoint-isolated canonical cache.")
    with open(manifest_path, encoding="utf-8") as handle:
        cache_signature = json.load(handle).get("cache_signature")
    if not isinstance(cache_signature, dict) or cache_signature.get("phase1_checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError("CLI --phase1_ckpt SHA256 does not match the probe cache manifest.")
    run_dir = os.path.abspath(output_dir)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    logs_dir = os.path.join(run_dir, "logs")
    predictions_dir = os.path.join(run_dir, "predictions")
    for directory in (run_dir, checkpoints_dir, logs_dir, predictions_dir):
        os.makedirs(directory, exist_ok=True)
    epoch_csv = os.path.join(logs_dir, "epoch_metrics.csv")
    epoch_jsonl = os.path.join(logs_dir, "epoch_metrics.jsonl")
    if task not in TASK_CACHE_KIND:
        raise ValueError(f"Unknown formal probe task {task!r}.")
    kind, cutoff = TASK_CACHE_KIND[task]
    active_splits = ("train", "val", "test") if evaluate_test else ("train", "val")
    print(f"[probe:{task}/{source}] opening {kind} cache for {','.join(active_splits)}...", flush=True)
    bases = {split: load_dataset(cache_root, kind, split=split, cutoff_min=cutoff) for split in active_splits}
    print(f"[probe:{task}/{source}] cache metadata ready; loading targets...", flush=True)
    targets, vocab = _targets_for_task(
        args.labels_csv, task, cfg, hcup_mapping_json=getattr(args, "hcup_mapping_json", None),
        allowed_splits=active_splits,
    )
    _validate_task_label_dimension(task, vocab, dict(cfg.get("tasks", {}).get(task, {})))
    active_ids = {hadm_id for split in active_splits for hadm_id in bases[split].admission_ids()}
    targets = {hadm_id: target for hadm_id, target in targets.items() if hadm_id in active_ids}
    smoke = bool(getattr(args, "smoke_mode", False))
    cap_names = {"train": "smoke_train_patients", "val": "smoke_val_patients", "test": "smoke_test_patients"}
    selected: Dict[str, List[int]] = {}; smoke_distributions: Dict[str, Dict[str, int]] = {}
    for offset, split in enumerate(active_splits):
        n = len(bases[split])
        cap = int(getattr(args, cap_names[split], n)) if smoke else n
        ids = bases[split].admission_ids()
        fixed_ids = getattr(args, "probe_selected_ids", {}).get(split) if getattr(args, "probe_selected_ids", None) else None
        if fixed_ids is not None:
            lookup = {hadm_id: index for index, hadm_id in enumerate(ids)}
            missing = [str(hadm_id) for hadm_id in fixed_ids if str(hadm_id) not in lookup]
            if missing:
                raise RuntimeError(f"Fixed probe cohort contains IDs absent from cache split={split}: {missing[:20]}")
            selected[split] = [lookup[str(hadm_id)] for hadm_id in fixed_ids]
            if len(selected[split]) != len(set(selected[split])):
                raise RuntimeError(f"Fixed probe cohort has duplicate IDs in split={split}.")
            smoke_distributions[split] = {"selected_patients": len(selected[split]), "fixed_cohort": 1}
        elif not smoke:
            selected[split] = list(range(n))
            smoke_distributions[split] = {"selected_patients": n}
        elif cap >= n:
            selected[split], smoke_distributions[split] = _smoke_indices(ids, targets, task, n, int(getattr(args, "seed", 42)) + offset, split)
        else:
            selected[split], smoke_distributions[split] = _smoke_indices(ids, targets, task, cap, int(getattr(args, "seed", 42)) + offset, split)
    train_cfg = dict(cfg["training"])
    datasets = {split: CanonicalProbeDataset(base, source, targets, selected[split]) for split, base in bases.items()}
    shard_cache_size = int(train_cfg.get("shard_cache_size", 2))
    for base in bases.values():
        if hasattr(base, "set_payload_cache_size"):
            base.set_payload_cache_size(shard_cache_size)
    print(f"[probe:{task}/{source}] cohort selected; preparing length-aware loaders...", flush=True)
    cohort_hash = hashlib.sha256(json.dumps({split: [bases[split].admission_ids()[i] for i in selected[split]] for split in selected}, sort_keys=True).encode()).hexdigest()
    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
    protocol_hash = str(getattr(args, "probe_protocol_hash", config_hash))
    dim = int(datasets["train"][0]["latent_seq"].size(-1))
    model = _make_task_model(task, dim, cfg, None if vocab is None else len(vocab["codes"]), source)
    device = torch.device(getattr(args, "device", "cuda") if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    num_workers = int(getattr(args, "num_workers", None)) if getattr(args, "num_workers", None) is not None else int(train_cfg.get("num_workers", 0))
    pin_memory = bool(train_cfg.get("pin_memory", device.type == "cuda"))
    persistent_workers = bool(train_cfg.get("persistent_workers", True)) and num_workers > 0
    prefetch_factor = train_cfg.get("prefetch_factor", 2)
    precision = str(train_cfg.get("precision", "32-true"))
    length_indices = {split: bases[split].sequence_lengths() for split in bases}
    runtime_autotune = None
    if bool(getattr(args, "autotune_probe_runtime", False)):
        runtime_autotune = _autotune_probe_runtime(
            model=model, dataset=datasets["train"], lengths=[length_indices["train"][i] for i in selected["train"]],
            shard_ids=datasets["train"].shard_indices(), task=task, cfg=cfg, train_cfg=train_cfg,
            device=device, seed=int(getattr(args, "seed", 42)), source=source, output_dir=run_dir,
            initial_num_workers=num_workers,
        )
        chosen = runtime_autotune["selected"]
        train_cfg["max_pairwise_elements_per_batch"] = int(chosen["max_pairwise_elements_per_batch"])
        train_cfg["max_patients_per_batch"] = int(chosen["max_patients_per_batch"])
        train_cfg["relative_query_chunk_size"] = int(chosen["relative_query_chunk_size"])
        num_workers = int(runtime_autotune["selected_num_workers"])
        train_cfg["prefetch_factor"] = int(runtime_autotune["selected_prefetch_factor"])
        persistent_workers = bool(train_cfg.get("persistent_workers", True)) and num_workers > 0
        prefetch_factor = train_cfg.get("prefetch_factor", 2)
    loaders = {}
    batch_samplers: Dict[str, Sampler[List[int]]] = {}
    for split, ds in datasets.items():
        if hasattr(ds, "shard_indices") and hasattr(bases[split], "shard_index"):
            sampler = ShardAwareAttentionCostBatchSampler(
                [length_indices[split][i] for i in selected[split]], ds.shard_indices(),
                int(train_cfg["max_pairwise_elements_per_batch"]), int(train_cfg["max_patients_per_batch"]),
                split == "train", int(getattr(args, "seed", 42)))
        else:
            sampler = AttentionCostBatchSampler(
                [length_indices[split][i] for i in selected[split]], int(train_cfg["max_pairwise_elements_per_batch"]),
                int(train_cfg["max_patients_per_batch"]), split == "train", int(getattr(args, "seed", 42)))
        batch_samplers[split] = sampler
        loader_kwargs: Dict[str, Any] = {"batch_sampler": sampler,
            "collate_fn": partial(canonical_probe_collate, validate=smoke), "num_workers": num_workers, "pin_memory": pin_memory,
            "persistent_workers": persistent_workers}
        if num_workers > 0 and prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
        loaders[split] = DataLoader(ds, **loader_kwargs)
    setup_seconds = time.perf_counter() - run_start
    print(
        f"[probe:{task}/{source}] device={device} precision={precision} "
        f"patients(train/val{'/test' if evaluate_test else ''})={len(datasets['train'])}/{len(datasets['val'])}"
        f"{('/' + str(len(datasets['test']))) if evaluate_test else ''} "
        f"batches(train/val{'/test' if evaluate_test else ''})={len(loaders['train'])}/{len(loaders['val'])}"
        f"{('/' + str(len(loaders['test']))) if evaluate_test else ''} "
        f"phase1_sha256={checkpoint_sha256}",
        flush=True,
    )
    print(f"[probe:{task}/{source}] provenance={json.dumps(provenance, sort_keys=True)}", flush=True)
    optimizer, optimizer_fused = _make_probe_optimizer(
        model.parameters(), device=device, lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]),
    )
    runtime_precision = resolve_runtime_precision(device, precision)
    scaler = _make_grad_scaler(device, runtime_precision)
    pos_weight = None
    if task in {"mortality", "los_ge_3d", "los_ge_7d", "readmission"}:
        base_train_ids = bases["train"].admission_ids()
        train_ids = [base_train_ids[index] for index in selected["train"]]
        y = np.asarray([float(targets[hadm_id]) for hadm_id in train_ids], dtype=np.float32)
        if not (0 < y.sum() < len(y)):
            raise RuntimeError("Binary probe training requires both positive and negative train examples.")
        with torch.no_grad():
            model.head.bias.fill_(float(np.log(y.mean() / (1.0 - y.mean()))))
        binary_cfg = dict(cfg["tasks"][task].get("binary_loss", {}))
        loss_type = str(binary_cfg.get("type", "unweighted_bce"))
        if loss_type not in {"unweighted_bce", "weighted_bce", "focal"}:
            raise ValueError(f"Unsupported binary_loss.type={loss_type!r}.")
        mode = str(binary_cfg.get("pos_weight_mode", "none"))
        if loss_type == "unweighted_bce" and mode != "none":
            raise ValueError("binary_loss.type=unweighted_bce requires pos_weight_mode=none.")
        raw_pos_weight = (len(y) - y.sum()) / y.sum()
        if mode == "none":
            pos_weight = None
        elif mode == "sqrt":
            pos_weight = torch.tensor(np.sqrt(raw_pos_weight), device=device)
        elif mode == "full":
            pos_weight = torch.tensor(raw_pos_weight, device=device)
        else:
            raise ValueError(f"Unsupported binary pos_weight_mode={mode!r}")
    start_epoch = 0; best_score, best_state, steps = -float("inf"), None, 0
    patience = int(train_cfg.get("early_stopping_patience", 0)); stale = 0; best_epoch = -1
    if getattr(args, "resume", None):
        resume_path = os.path.realpath(args.resume)
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        required = {"model_state", "optimizer_state", "epoch", "global_step", "best_score", "best_epoch", "stale_epochs", "rng_state", "phase1_checkpoint_sha256"}
        if (not isinstance(payload, dict) or required.difference(payload)
                or payload["phase1_checkpoint_sha256"] != checkpoint_sha256
                or payload.get("cache_signature") != cache_signature
                or payload.get("cohort_hash") != cohort_hash or payload.get("config_hash") != config_hash
                or payload.get("protocol_hash") != protocol_hash):
            raise RuntimeError("Probe resume checkpoint is incomplete or bound to a different Phase-1 checkpoint.")
        model.load_state_dict(payload["model_state"], strict=True); optimizer.load_state_dict(payload["optimizer_state"])
        _restore_rng_state(payload["rng_state"]); start_epoch = int(payload["epoch"]) + 1; steps = int(payload["global_step"])
        best_score = float(payload["best_score"]); best_epoch = int(payload["best_epoch"]); stale = int(payload["stale_epochs"])
        best_resume_path = os.path.join(os.path.dirname(resume_path), "best.pt")
        if not os.path.isfile(best_resume_path):
            raise RuntimeError("Probe resume requires sibling checkpoints/best.pt.")
        best_state = torch.load(best_resume_path, map_location="cpu", weights_only=False)["model_state"]
        if not os.path.isfile(epoch_csv):
            raise RuntimeError("Probe resume requires the existing epoch_metrics.csv for duplicate-epoch protection.")
        logged = pd.read_csv(epoch_csv)
        if logged.empty or int(logged["epoch"].max()) != int(payload["epoch"]):
            raise RuntimeError("Probe resume checkpoint epoch does not match the persisted epoch log.")
    train_seconds = 0.0; validation_seconds = 0.0; early_stop_reason = "max_epochs"
    for epoch in range(start_epoch, int(train_cfg["epochs"])):
        epoch_start = time.perf_counter()
        loaders["train"].batch_sampler.set_epoch(epoch)
        consistency_weight = 0.0
        if bool(cfg["density_robustness"].get("enabled", False)):
            warmup = int(cfg["density_robustness"].get("consistency_warmup_epochs", 0))
            ramp = int(cfg["density_robustness"].get("consistency_ramp_epochs", 0))
            target_weight = float(cfg["density_robustness"].get("consistency_weight", 0.0))
            consistency_weight = 0.0 if epoch < warmup else target_weight * min(1.0, (epoch - warmup + 1) / max(ramp, 1))
        epoch_density_cfg = dict(cfg["density_robustness"])
        epoch_density_cfg["effective_consistency_weight"] = consistency_weight
        epoch_density_cfg["finite_check_interval"] = int(train_cfg.get("finite_check_interval", 0))
        epoch_density_cfg["binary_loss_type"] = (str(cfg["tasks"][task].get("binary_loss", {}).get("type", "unweighted_bce"))
                                                  if task in {"mortality", "los_ge_3d", "los_ge_7d", "readmission"} else "unweighted_bce")
        train_telemetry: Dict[str, Any] = {}
        train_start = time.perf_counter()
        train_loss, _, _, _ = _epoch(model, loaders["train"], task, device, optimizer, epoch_density_cfg, collect_predictions=False,
                                      pos_weight=pos_weight, gradient_clip_val=train_cfg.get("gradient_clip_val"), precision=precision,
                                      strict_finite=smoke, progress_desc=f"probe:{task}/{source} epoch {epoch + 1} train",
                                      effective_batch_patients=int(train_cfg.get("effective_batch_patients", 1)), telemetry=train_telemetry,
                                      scaler=scaler)
        train_seconds += time.perf_counter() - train_start
        steps += int(train_telemetry["optimizer_steps"])
        validation_start = time.perf_counter()
        val_density_cfg = dict(cfg["density_robustness"])
        val_density_cfg["finite_check_interval"] = int(train_cfg.get("finite_check_interval", 0))
        val_loss, _, vy, vz = _epoch(model, loaders["val"], task, device, None, val_density_cfg, precision="32-true",
                                      strict_finite=smoke, progress_desc=f"probe:{task}/{source} epoch {epoch + 1} val")
        validation_seconds += time.perf_counter() - validation_start
        score, val_metrics = _task_metric(task, vy, vz)
        peak_gb = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if device.type == "cuda" else 0.0
        val_metric_text = " ".join(
            f"{key}={value:.6f}" if np.isfinite(value) else f"{key}=nan"
            for key, value in sorted(_scalar_metric_fields("", val_metrics).items())
        )
        print(
            f"[probe:{task}/{source}] epoch={epoch + 1}/{int(train_cfg['epochs'])} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} monitor={score:.6f} best={best_score:.6f} "
            f"epoch_sec={time.perf_counter() - epoch_start:.1f} gpu_peak_gb={peak_gb:.2f} "
            f"val_metrics: {val_metric_text}",
            flush=True,
        )
        improved = score > best_score + float(train_cfg.get("early_stopping_min_delta", 0.0))
        if improved:
            best_score, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch, stale = epoch, 0
            torch.save({"architecture_version": PROBE_ARCHITECTURE_VERSION, "task": task, "source_name": source,
                        "encoder": cfg["encoder"], "latent_dim": dim, "label_vocab": vocab,
                        "state_dict": best_state, "model_state": best_state, "epoch": epoch, "best_score": best_score,
                        "phase1_checkpoint_sha256": checkpoint_sha256, "phase1_checkpoint_path": resolved_ckpt,
                        "cache_schema_version": cache_signature.get("cache_schema_version"),
                        "schema_sha256": cache_signature.get("schema_sha256"), "cache_signature": cache_signature,
                        "vocab_hash": _vocab_hash(vocab) if vocab else None, "config": cfg,
                        "cohort_hash": cohort_hash, "config_hash": config_hash, "protocol_hash": protocol_hash,
                        "provenance": provenance}, os.path.join(checkpoints_dir, "best.pt"))
        else:
            stale += 1
        last_payload = {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "optimizer_state": optimizer.state_dict(), "epoch": epoch, "global_step": steps,
                        "best_score": best_score, "best_epoch": best_epoch, "stale_epochs": stale,
                        "rng_state": _rng_state(), "task": task, "phase1_checkpoint_sha256": checkpoint_sha256,
                        "cache_signature": cache_signature, "config": cfg, "cohort_hash": cohort_hash,
                        "config_hash": config_hash, "protocol_hash": protocol_hash, "provenance": provenance}
        torch.save(last_payload, os.path.join(checkpoints_dir, "last.pt"))
        _write_epoch_log(epoch_csv, epoch_jsonl, {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "monitor": score, "best_score": best_score, "best_epoch": best_epoch, "stale_epochs": stale,
            "optimizer_steps": steps, "epoch_seconds": time.perf_counter() - epoch_start,
            "effective_consistency_weight": consistency_weight, "learning_rate": optimizer.param_groups[0]["lr"],
            **_scalar_metric_fields("val_", val_metrics),
            **train_telemetry})
        if not improved and epoch + 1 >= int(train_cfg.get("min_epochs", 1)) and patience and stale >= patience:
            early_stop_reason = "early_stopping_patience"
            break
        if int(getattr(args, "max_steps", -1)) > 0 and steps >= int(args.max_steps):
            early_stop_reason = "max_steps"
            break
    if best_state is None:
        raise RuntimeError("Probe training produced no validation checkpoint.")
    model.load_state_dict(best_state, strict=True)
    validation_start = time.perf_counter()
    _, val_ids, val_target, val_logits = _epoch(model, loaders["val"], task, device, None, cfg["density_robustness"], precision="32-true",
                                                 strict_finite=smoke, progress_desc=f"probe:{task}/{source} final validation")
    validation_seconds += time.perf_counter() - validation_start
    thresholds = _fit_multilabel_thresholds(val_target, val_logits, vocab) if task in {"icd", "hcup"} else None
    best_path = os.path.join(checkpoints_dir, "best.pt")
    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    best_payload.update({"thresholds": thresholds, "pos_weight": None if pos_weight is None else float(pos_weight.cpu()),
                         "full_train_positive_count": int(y.sum()) if task in {"mortality", "los_ge_3d", "los_ge_7d", "readmission"} else None,
                         "full_train_negative_count": int(len(y) - y.sum()) if task in {"mortality", "los_ge_3d", "los_ge_7d", "readmission"} else None,
                         "binary_loss": dict(cfg["tasks"][task].get("binary_loss", {})) if task in {"mortality", "los_ge_3d", "los_ge_7d", "readmission"} else None})
    torch.save(best_payload, best_path)
    save_predictions = bool(getattr(args, "save_predictions", False))
    if save_predictions:
        _write_prediction_npz(os.path.join(predictions_dir, "val_best.npz"), task, val_ids, val_target, val_logits)
    metrics: Dict[str, float] = {}
    test_seconds = 0.0; reload_test_seconds = 0.0; ids: List[str] = []
    reload_probability_diff = None; reload_decode_equal = None; reload_max_abs_logit_diff = None
    if evaluate_test:
        test_start = time.perf_counter()
        _, ids, ty, tz = _epoch(model, loaders["test"], task, device, None, cfg["density_robustness"], precision="32-true",
                                 strict_finite=smoke, progress_desc=f"probe:{task}/{source} test")
        test_seconds = time.perf_counter() - test_start
        _, metrics = _task_metric(task, ty, tz)
        if thresholds is not None:
            metrics.update(multilabel_threshold_metrics(ty, tz, thresholds))
        # Reload an entirely new model from the persisted best artifact.
        del model
        restored = _make_task_model(task, dim, cfg, None if vocab is None else len(vocab["codes"]), source).to(device).eval()
        restored.load_state_dict(best_payload["state_dict"], strict=True)
        reload_test_start = time.perf_counter()
        _, restored_ids, restored_target, restored_logits = _epoch(restored, loaders["test"], task, device, None, cfg["density_robustness"], precision="32-true",
                                                                      strict_finite=smoke, progress_desc=f"probe:{task}/{source} reload test")
        reload_test_seconds = time.perf_counter() - reload_test_start
        if restored_ids != ids or not torch.equal(restored_target, ty):
            raise RuntimeError("Reloaded probe test IDs/targets do not strictly align.")
        if task == "drg":
            logits_match = (torch.allclose(tz[0], restored_logits[0], atol=1e-5, rtol=1e-4)
                            and torch.allclose(tz[1], restored_logits[1], atol=1e-5, rtol=1e-4))
        else:
            logits_match = torch.allclose(tz, restored_logits, atol=1e-5, rtol=1e-4)
        if not logits_match:
            raise RuntimeError("Reloaded probe logits do not match the saved checkpoint prediction.")
        if task == "drg":
            reload_probability_diff = float(max((_probabilities(task, tz)[0] - _probabilities(task, restored_logits)[0]).abs().max(),
                                                (_probabilities(task, tz)[1] - _probabilities(task, restored_logits)[1]).abs().max()).item())
            reload_decode_equal = bool(torch.equal(CompetitiveDRGProbe.decode(tz[0], tz[1]), CompetitiveDRGProbe.decode(restored_logits[0], restored_logits[1])))
        else:
            reload_probability_diff = float((_probabilities(task, tz) - _probabilities(task, restored_logits)).abs().max().item())
            reload_decode_equal = None
        _, reload_metrics = _task_metric(task, restored_target, restored_logits)
        if thresholds is not None:
            reload_metrics.update(multilabel_threshold_metrics(restored_target, restored_logits, thresholds))
        if reload_metrics != metrics:
            raise RuntimeError("Reloaded probe metrics do not match the original test evaluation.")
        if save_predictions:
            _write_prediction_npz(os.path.join(predictions_dir, "test.npz"), task, ids, ty, tz)
    max_length = max(max(length_indices[split][i] for i in selected[split]) for split in selected)
    max_batch_cost = max(
        len(batch) * max(length_indices["train"][selected["train"][i]] for i in batch) ** 2
        for batch in loaders["train"].batch_sampler
    )
    all_lengths = np.asarray([length_indices[split][index] for split in selected for index in selected[split]], dtype=np.float64)
    manifest = {"task": task, "source_name": source, "smoke_mode": smoke, "epochs_completed": epoch + 1,
                "optimizer_steps": steps, "max_sequence_length": max_length, "max_batch_cost": max_batch_cost,
                "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0),
                "runtime_seconds": float(time.perf_counter() - run_start), "requested_precision": precision,
                "effective_precision": runtime_precision,
                "num_workers": num_workers, "pin_memory": pin_memory, "persistent_workers": persistent_workers,
                "prefetch_factor": prefetch_factor if num_workers > 0 else None, "smoke_label_distribution": smoke_distributions,
                "setup_seconds": setup_seconds, "train_seconds": train_seconds, "validation_seconds": validation_seconds,
                "test_seconds": test_seconds, "reload_test_seconds": reload_test_seconds,
                "train_patients_per_second": float(train_telemetry.get("patients", 0) / max(train_seconds, 1e-12)),
                "test_patients_per_second": float(len(ids) / max(test_seconds, 1e-12)) if evaluate_test else None,
                "sequence_length_p50": float(np.percentile(all_lengths, 50)), "sequence_length_p90": float(np.percentile(all_lengths, 90)),
                "sequence_length_p95": float(np.percentile(all_lengths, 95)), "sequence_length_p99": float(np.percentile(all_lengths, 99)),
                "last_epoch_microbatch_sizes": train_telemetry.get("microbatch_sizes", []),
                "last_epoch_accumulation_cycle_patients": train_telemetry.get("accumulation_cycle_patients", []),
                "last_epoch_finite_checks": train_telemetry.get("finite_checks", 0),
                "early_stop_reason": early_stop_reason, "training_precision": precision,
                "evaluation_precision": "32-true", "best_epoch": best_epoch, "best_score": best_score,
                "optimizer_fused": optimizer_fused,
                "finite_check_mode": "strict" if smoke else "epoch",
                "finite_check_interval": 1 if smoke else 0,
                "shard_cache_size": shard_cache_size,
                "runtime_autotune": runtime_autotune,
                "shard_sampler": {
                    split: {
                        "type": type(batch_samplers[split]).__name__,
                        "batches_per_epoch": len(batch_samplers[split]),
                        "shard_switch_count": int(getattr(batch_samplers[split], "shard_switches", 0)),
                        "average_shards_per_batch": 1.0,
                        "selected_shard_count": len(set(getattr(batch_samplers[split], "shard_ids", [0]))),
                        "estimated_shard_loads_per_epoch": len(set(getattr(batch_samplers[split], "shard_ids", [0]))),
                    } for split in batch_samplers
                },
                "shard_payload_stats": {
                    split: (bases[split].shard_stats() if hasattr(bases[split], "shard_stats") else {})
                    for split in bases
                },
                "reload_max_abs_logit_diff": (float(max((tz[0] - restored_logits[0]).abs().max(), (tz[1] - restored_logits[1]).abs().max()).item()) if task == "drg" else float((tz - restored_logits).abs().max().item())) if evaluate_test else None,
                "reload_probability_diff": reload_probability_diff, "reload_metrics_equal": True,
                "reload_decode_equal": reload_decode_equal,
                "provenance": provenance, "cache_signature": cache_signature}
    with open(os.path.join(output_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    with open(os.path.join(output_dir, "final_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(({"test": metrics, "n_test": len(ids)} if evaluate_test else {"validation_primary_metric": best_score}), handle, indent=2)
    complete_tmp = os.path.join(output_dir, f".complete.tmp.{os.getpid()}")
    with open(complete_tmp, "w", encoding="utf-8") as handle:
        handle.write("complete\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(complete_tmp, os.path.join(output_dir, ".complete"))
    result = {"checkpoint": best_path, "task": task, "source_name": source,
              "validation_primary_metric": best_score, "best_epoch": best_epoch}
    if evaluate_test:
        result["test_metrics"] = metrics
    return result
