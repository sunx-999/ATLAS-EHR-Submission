# ATLAS-EHR

ATLAS-EHR is research code for learning and evaluating longitudinal representations of structured electronic health record (EHR) events. The primary development workflow uses MIMIC-IV-derived, already prepared event buckets; eICU is supported for hospital-admission-level external validation.

The repository contains a Phase-1 JEPA-style representation learner, frozen-representation caches, exact-time downstream probes, numeric readout and imputation analyses, state-only baselines, robustness and ablation utilities, and a separate Hugging Face release package.

## Overview

The central workflow is:

1. Pack already exported MIMIC-IV event buckets into admission-level tensors.
2. Train the Phase-1 model with a masked-numeric prediction objective.
3. Build canonical frozen representation caches from an exact Phase-1 checkpoint.
4. Train and evaluate downstream probes over those caches.
5. Run external validation, baselines, robustness checks, ablations, or post-training analyses as needed.

ATLAS-EHR is research software. It is not a clinical device and this repository does not claim clinical validation or superiority over other systems.

## Model at a glance

| Component | Current implementation |
| --- | --- |
| Input representation | Timestamped sets of state/event VIDs with values, padding masks, and admission-relative time. `schema_processed.csv` supplies feature identity and type information. |
| Numeric handling | Numeric state features use a dedicated value/severity branch; Phase-1 masking is restricted to numeric state events. |
| `z_event` | Event-level EMA-teacher target for packed masked numeric queries. |
| `z_set` | Per-timestamp state-set representation produced by the set encoder. |
| `z_context` | Temporally contextualized per-timestamp representation built from `z_set` by the relative-time backbone. |
| Student and teacher | The student encodes a masked context. The target family is an EMA teacher comprising the feature projector, fusion normalization, set encoder, temporal backbone, and context output normalization. |
| Canonical probe sources | Exact-time downstream probes consume `z_set` and `z_context`. |

The current Phase-1 architecture identifier is stored in checkpoints and used by cache and artifact compatibility checks.

## Repository structure

```text
src/
  mimic_prep.py                    MIMIC bucket packer and schema/preprocessing artifacts
  eicu_prep.py                     eICU hospital-admission preprocessing; legacy ICU-stay path
  dataset.py                        Dataset, sequence policy, and collation contracts
  model.py                          Phase-1 JEPA model and EMA teacher
  train.py                          Phase-1, cache, readout, and probe-training entry point
  cache_bank.py                     Canonical representation and readout-query cache contracts
  probes.py                         Exact-time downstream probe architectures and training helpers
  eval.py                           Formal MIMIC frozen-probe evaluation
  readout.py                        Target-manifold numeric readout and evaluation utilities
  benchmark.py                      State-only logistic-regression and XGBoost baselines
  robustness.py                     Raw-event MCAR robustness evaluation
  ablation.py                       Masked-numeric temporal-context (H/C/F) ablation
  external_validation.py            Canonical eICU external-validation workflows
  foundation_validation_pipeline.py Resumable validation orchestration
  analysis.py                       Backward-compatible analysis CLI/router facade
  analysis_set_encoder.py           Set-encoder attention analysis implementation
  analysis_visualization.py         Representation and paper-oriented visualization implementation
  analysis_icd.py                   ICD probe-attribution implementation
  analysis_imputation.py            Reconstruction/imputation analysis implementation

configs/params.yaml                Phase-1, readout, probe, benchmark, and shared settings
BigQuery/                           Reference SQL and related preprocessing material
scripts/                            Result-figure, table, and audit utilities
tests/                              Contract and regression tests
hf_release/ATLAS-EHR/               Separate public-runtime release package and artifacts
```

`src.analysis` intentionally contains no scientific analysis implementation. It preserves the historical CLI and qualified Python attributes while routing to the four analysis modules listed above.

## Installation

The research repository currently has no root-level `pyproject.toml`, `requirements.txt`, or environment lockfile. Create an environment that provides the libraries imported by the selected workflow before running its CLI. The local development environment used by this repository is `.venv`.

The packaged Hugging Face runtime is separate from the research code and has its own dependency declaration under [`hf_release/ATLAS-EHR/`](hf_release/ATLAS-EHR/).

```bash
git clone https://huggingface.co/sunx999/ATLAS-EHR
cd ATLAS-EHR
python -m pip install .
```

That installation command applies to the release package, not to the root research workflow.

## Data preparation

### MIMIC-IV

`src/mimic_prep.py` is the MIMIC preprocessing entry point. It is a packer for already exported bucketed JSONL data; it does not directly extract raw MIMIC-IV relational tables. The repository also contains MIMIC-oriented reference material under `BigQuery/MIMIC-IV V3.1/`.

The packer expects:

- `--buckets_dir`: split/bucket directories containing JSONL shards;
- `--schema_csv`: a `schema_processed.csv`-style feature schema;
- `--labels_csv`: admission labels and split assignments;
- `--output_dir`: destination root;
- optional `--freq_csv`: feature-frequency input; and
- optional `--num_workers`: parallel worker count.

```bash
python -m src.mimic_prep \
  --buckets_dir /path/to/mimic_buckets \
  --schema_csv /path/to/schema_processed.csv \
  --labels_csv /path/to/labels.csv \
  --output_dir /path/to/mimic_processed \
  --freq_csv /path/to/states_frequency.csv \
  --num_workers 4
```

The output contract includes one packed `.pt` sample per admission at `output_dir/{split}/{hadm_id}.pt`, `schema.json`, and `prep_summary.json`. The summary filename remains `prep_summary.json`.

### eICU

`src/eicu_prep.py` prepares eICU trajectories for external validation. The formal default is:

```text
--sample_unit hospital_admission
```

Its canonical formal identifier is `patienthealthsystemstayid`. The path aggregates ICU stays into a hospital-admission sample and records the formal sample-unit/time-origin contract in `prep_summary.json`.

```bash
python -m src.eicu_prep \
  --input_dir /path/to/eicu_export \
  --schema_csv /path/to/schema_processed.csv \
  --labels_json /path/to/labels.json \
  --output_dir /path/to/eicu_processed \
  --sample_unit hospital_admission
```

`--sample_unit icu_stay` is retained for historical reproduction only. It is not the formal external-validation path. Likewise, `--labels_only` is explicitly legacy-only and rejected for formal validation.

## Phase-1 training

`configs/params.yaml` is the configuration source for Phase-1 architecture, data policy, recipe, normalization, training, and performance settings. `src/train.py` accepts the paths below and writes its run under `--output_dir` and `--run_name`.

```bash
python -m src.train \
  --phase 1 \
  --config configs/params.yaml \
  --data_dir /path/to/mimic_processed \
  --schema_dir /path/to/schema_directory \
  --labels_csv /path/to/labels.csv \
  --output_dir /path/to/outputs \
  --run_name phase1_run \
  --devices 1 \
  --accelerator cuda
```

Use a device/accelerator setting appropriate for the host. GPU execution is generally appropriate for training, but local CPU-safe tests and smoke workflows are also supported by the codebase.

## Frozen representation cache

`src/cache_bank.py` owns canonical cache layouts and validation. For exact-time probe evaluation, the canonical latent sources are:

- `z_set`
- `z_context`

The cache stages are:

| Stage | Meaning | Cutoff |
| --- | --- | ---: |
| `full_visible` | Entire visible admission sequence | none |
| `retrospective_cutoff` | Information available before a task cutoff | 24 h or 48 h |
| `readout_training_cache` | Numeric target-manifold query cache | task-specific query contract |

The canonical retrospective cutoffs are 1440 minutes (24 h) and 2880 minutes (48 h). Build an exact Phase-1 probe cache through the training entry point:

```bash
python -m src.train \
  --phase probe_cache_build \
  --config configs/params.yaml \
  --data_dir /path/to/mimic_processed \
  --schema_dir /path/to/schema_directory \
  --labels_csv /path/to/labels.csv \
  --phase1_ckpt /path/to/phase1.ckpt \
  --cache_dir /path/to/cache_root \
  --output_dir /path/to/outputs \
  --run_name probe_cache
```

Cache and artifact validation binds checkpoint identity/hash, schema identity/hash, architecture version, dataset/config signatures, cache schema/contract versions, and stage/split visibility. Rebuild a cache when a required contract check reports incompatibility.

## Downstream probes

`src/probes.py` defines the exact-time probe family; `src.eval.py` evaluates frozen probes against canonical caches. Formal task-to-visibility mapping is defined by `TASK_CACHE_KIND`.

| Task | Visibility/cache mode | Endpoint type |
| --- | --- | --- |
| `mortality` | retrospective cutoff, 2880 min (48 h) | binary |
| `los_ge_3d` | retrospective cutoff, 1440 min (24 h) | binary |
| `los_ge_7d` | retrospective cutoff, 1440 min (24 h) | binary |
| `readmission` | full visible | binary |
| `icd` | full visible | multilabel |
| `hcup` | full visible | multilabel |
| `drg` | full visible | DRG classification |

Train a selected task/source through `src.train` after building the matching cache:

```bash
python -m src.train \
  --phase probe_train \
  --config configs/params.yaml \
  --data_dir /path/to/mimic_processed \
  --schema_dir /path/to/schema_directory \
  --labels_csv /path/to/labels.csv \
  --phase1_ckpt /path/to/phase1.ckpt \
  --cache_dir /path/to/cache_root \
  --output_dir /path/to/outputs \
  --run_name probes \
  --probe_task mortality \
  --probe_source z_set
```

For formal evaluation, provide either one `--probe_checkpoint` or a `--probe_root`:

```bash
python -m src.eval \
  --probe_root /path/to/probe_root \
  --phase1_ckpt /path/to/phase1.ckpt \
  --cache_root /path/to/cache_root \
  --labels_csv /path/to/labels.csv \
  --config configs/params.yaml \
  --output_dir /path/to/evaluation \
  --tasks all \
  --sources all
```

The evaluator checks probe, cache, checkpoint, schema, source, task, and visibility compatibility before producing formal outputs.

## Numeric readout and imputation analysis

These are distinct layers:

- `src/readout.py` implements the non-parametric target-manifold numeric readout. It decodes numeric values by inverting the frozen EMA-teacher event manifold and also contains streaming evaluation and per-VID ridge comparison utilities.
- `src/analysis_imputation.py` implements downstream reconstruction/imputation analyses and figures using the readout/query contracts. It is not the readout-training implementation itself.

The readout workflow is available through `src.train`:

```bash
python -m src.train \
  --phase readout_probe \
  --config configs/params.yaml \
  --data_dir /path/to/mimic_processed \
  --schema_dir /path/to/schema_directory \
  --labels_csv /path/to/labels.csv \
  --phase1_ckpt /path/to/phase1.ckpt \
  --cache_dir /path/to/cache_root \
  --output_dir /path/to/outputs \
  --run_name readout
```

For imputation-analysis arguments, use the analysis router help rather than copying an exploratory command into a formal workflow:

```bash
python -m src.analysis imputation --help
```

## External validation

`src/external_validation.py` runs canonical eICU external-validation workflows against a MIMIC-trained Phase-1 checkpoint. The CLI subcommands are:

| Command | Purpose |
| --- | --- |
| `preflight` | Check required eICU inputs and formal alignment prerequisites. |
| `cache_build` | Build the external canonical cache. |
| `reuse_pretrained` | Evaluate a MIMIC-trained probe on the external cache. |
| `retrain_probe` | Train a local external probe through the canonical retraining implementation. |
| `evaluate_retrained` | Evaluate a retrained external probe and optionally combine results. |

All formal eICU paths are hospital-admission-level. The external validator rejects legacy ICU-stay caches and requires the eICU preprocessing metadata to identify `patienthealthsystemstayid` and a hospital-admission time origin.

Discover command-specific requirements with:

```bash
python -m src.external_validation --help
```

## Baselines and validation utilities

| Module | Role |
| --- | --- |
| `src/benchmark.py` | State-only baselines using logistic regression (`lr`) and XGBoost (`xgb`). Available tabular summaries are `last`, `mean`, `max`, `presence`, `count`, and `interval`. |
| `src/robustness.py` | Raw-event MCAR robustness evaluation for canonical exact-time probes across configured drop rates and repeats. |
| `src/ablation.py` | Phase-1 masked-numeric temporal-context ablation over query-only, history, current, future, and their H/C/F combinations; it uses a frozen readout artifact rather than fitting a new decoder. |
| `src/foundation_validation_pipeline.py` | Resumable orchestration of cache, probe, evaluation, baseline, external-validation, statistical-analysis, and finalization stages. It delegates scientific implementations to canonical modules. |

Use each module's CLI help to discover the full contract:

```bash
python -m src.benchmark --help
python -m src.robustness --help
python -m src.ablation --help
python -m src.foundation_validation_pipeline --help
```

## Analysis tools

The supported analysis modes are exposed through the stable facade:

```bash
python -m src.analysis <mode> --help
```

| Mode | Implementation module | Scope |
| --- | --- | --- |
| `set_encoder_attention` | `src.analysis_set_encoder` | Local set-encoder attention analysis with withheld target variables. |
| `icd_probe_attribution` | `src.analysis_icd` | Exact-time ICD attribution and cohort analyses. |
| `imputation` | `src.analysis_imputation` | Readout-output reconstruction/imputation analyses. |
| `visualize` | `src.analysis_visualization` | Latent, temporal, and report-oriented visualization workflows. |

The four modes are the current `src.analysis.MODES`. Their flags are intentionally mode-specific; use the help command above rather than assuming a shared analysis argument set.

## Configuration

[`configs/params.yaml`](configs/params.yaml) contains the current top-level configuration domains:

| Section | Covers |
| --- | --- |
| `phase1` | Model, data, recipe, normalization, training, and performance settings. |
| `readout` | Target-manifold architecture, support, calibration/search, oracle, and cache settings. |
| `probe` | Probe architecture, cache construction, encoder, robustness, training, and task settings. |
| `benchmark` | Feature-builder, runtime, linear, and XGBoost settings. |
| `shared` | Shared numerical settings. |

CLI arguments may select paths and operational settings, while architecture and scientific defaults remain configuration-driven. Do not substitute a schema or checkpoint from another run without satisfying the corresponding compatibility checks.

## Reproducibility and provenance

Formal paths record and validate provenance rather than silently reusing incompatible artifacts. Depending on the workflow, safeguards include:

- checkpoint and schema real paths and SHA256 identities;
- Phase-1 architecture identity and latent dimension;
- cache schema and cache-contract versions;
- dataset-content, masking, and configuration signatures;
- split, cutoff, task, source, and exact-time alignment checks; and
- artifact-specific compatibility metadata.

These checks support traceability of a configured run; they do not by themselves guarantee complete reproducibility across data releases, hardware, or software environments.

## Hugging Face release

The public runtime and release artifacts are maintained under [`hf_release/ATLAS-EHR/`](hf_release/ATLAS-EHR/). Its checked-in release documentation identifies the public repository as [`sunx999/ATLAS-EHR`](https://huggingface.co/sunx999/ATLAS-EHR) and describes a safetensors/JSON runtime that is separate from Lightning checkpoints and the research `src/` package.

See the release package README for its prepared-input requirements, schema contract, and inference API. The research repository's preprocessing and training workflows should not be conflated with the public runtime API.

## Citation

A citation will be added upon publication.

## License

This repository is licensed under the [Apache License 2.0](LICENSE).

## CLI reference checks

The following current entry points expose their parser contract through `--help`:

```bash
python -m src.mimic_prep --help
python -m src.eicu_prep --help
python -m src.train --help
python -m src.eval --help
python -m src.external_validation --help
python -m src.benchmark --help
python -m src.robustness --help
python -m src.ablation --help
python -m src.foundation_validation_pipeline --help
python -m src.analysis set_encoder_attention --help
python -m src.analysis icd_probe_attribution --help
python -m src.analysis imputation --help
python -m src.analysis visualize --help
```
