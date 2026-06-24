# Robust_Rerun - leakage-free re-run of the Paper 6 graph contribution

One-stop, resume-capable pipeline that rebuilds the Paper 6 (Multilingual Stereotype
Knowledge Graph) results under a strict, leakage-free protocol, so the contribution
survives a hostile methods review. Read this file top to bottom and you can run the
whole study on one GPU VM. No prior pipeline output is overwritten: everything lands
under `Robust_Rerun/results/`.

It reuses the verified Paper 5 cut compute and the shared `Common_00` library that
already sit in `Codes/paper6/`; it does not duplicate them.

---

## 1. What this fixes (and why)

The original Paper 6 had three defects a reviewer rejects on. Each is removed here.

| Defect (original) | Fix (this pipeline) |
|-------------------|---------------------|
| The KG was built from the **entire evaluation set**, so the propagation 0.767 was circular and retrieval was tested on the items that defined the graph. | **Concept-level train/val/test split.** The graph is built from TRAIN concepts only; retrieval is evaluated on HELD-OUT TEST concepts; calibration uses VAL concepts. Test/val groups never enter the graph. |
| Graph-RAG injected the query item's **own anti-stereotype target** (the gold answer), which is why it over-corrected. | Retrieval draws counter-evidence only from TRAIN concepts and **excludes the query item's own group surfaces and concept**. |
| The "graph bridge" never used the graph (it reused a target-language cut), and residual units are not graph nodes. | The in-parameter "bridge" headline is dropped. The graph's real intervention is **leakage-free retrieval**, evaluated against proper baselines. The disjointness diagnosis is re-run with **5 seeds + a random-mask null**. The propagation claim is replaced by a **held-out same_as link-prediction** test vs a random-relabel null. |

The defensible, leakage-proof contribution this pipeline measures: a **weight-free,
cross-lingual, inspectable** retrieval intervention that runs on **black-box API
models** and is validated on **held-out concepts** against generic-prompt,
translate-to-English, and flat-dictionary baselines.

---

## 2. Folder layout and execution order

```
Robust_Rerun/
  run_all.py                  single entry point (resume-capable)
  config/default.yaml         all NON-SECRET config: paths, models, providers, hyperparameters
  .env.example                key NAMES only; values load from the repo-root .env
  requirements.txt            global install (no venv)
  install_flash_attention.py  pre-compiled flash-attn wheel (run once on the GPU VM)
  lib/
    paths.py        config access + sys.path wiring
    integrity.py    duplicate / corruption check (runs every rerun)
    splits.py       concept-level train/val/test split (union-find over same_as)
    kg_build.py     build the MS-SKG from TRAIN concepts only
    rag_leakfree.py leakage-free retrieval + all baselines/ablations
    stats.py        bootstrap CI, Cohen's d, Jaccard, Cochran's Q / I-squared
  step_01_integrity.py     [cpu]  duplicate/corruption guard (hard-fails on bad data)
  step_02_split.py         [cpu]  concept_split.json
  step_03_build_kg.py      [cpu]  MS-SKG from train concepts
  step_04_dry_run.py       [api]  test every provider key + subject model id + flash-attn
  step_05_cut_stability.py [gpu]  5-seed minimal cut (for the disjointness null)
  step_06_propagation.py   [cpu]  held-out same_as link prediction vs null
  step_07_calibrate.py     [api]  pick retrieval n_facts on VAL concepts (remove over-correction)
  step_08_rag_local.py     [gpu]  leakage-free RAG + baselines, local decoders, TEST concepts
  step_09_rag_api.py       [api]  leakage-free RAG + baselines, black-box subjects, TEST concepts
  step_10_analysis.py      [cpu]  aggregate -> summary.json + forest_plot.csv
  tests/
    test_no_secrets.py     guard: no hard-coded credential anywhere in this tree
    test_core.py           split determinism, retrieval self-exclusion, stats
  results/                 all outputs (created at runtime)
```

Run order is the numeric step suffix. `run_all.py` always runs step 01 first.

---

## 3. Setup (global environment, no venv)

```bash
cd Codes/paper6/Robust_Rerun
pip install -r requirements.txt
```

Do not create a venv; install into the global interpreter.

### Flash-attention (important for GPU speed)

Flash-attention-2 makes the local decoders much faster. Install only a **pre-compiled
wheel** (never build from source), and only on the GPU VM (Linux x86_64 + CUDA):

```bash
python install_flash_attention.py            # auto-detect torch/CUDA/python/ABI
python install_flash_attention.py --version 2.8.3
```

The script detects torch, CUDA, the Python tag, and the C++11 ABI, constructs the
matching `Dao-AILab/flash-attention` release wheel URL, verifies it exists, then
`pip install`s it with `--no-build-isolation`. On non-Linux / non-CUDA hosts it skips
cleanly and the decoders fall back to SDPA attention (slower, still correct).
`config.backbone.use_flash_attention: true` makes the loader request it when present.

The OS must be compatible: pre-built flash-attn wheels are Linux x86_64 only.

---

## 4. Secrets and judge providers (.env)

Only secrets live in `.env` (repo root, `Codes/paper6/.env`, git-ignored). base_url and
model ids are NOT secrets and live in `config/default.yaml`. No key value ever appears in
any tracked file (enforced by `tests/test_no_secrets.py`).

Judge / answer-extraction (config `judge`):

| Role | Provider | Model | Key env-var(s) |
|------|----------|-------|----------------|
| PRIMARY | LinkAPI | `gemini-2.5-pro` | `Link_Gemini_Cheap_API_Key` |
| SECONDARY | DeepSeek | `deepseek-chat` | `DEEPSEEK_API_KEY_1`, `DEEPSEEK_API_KEY_2` |
| TERTIARY | Mistral | `mistral-small-latest` | `MISTRAL_API_KEY1`, `MISTRAL_API_KEY2` |

- The active provider is fixed by `judge.provider`. **There is NO automatic cross-provider
  fallback**: a failed call is recorded as a failure, never silently retried on another
  provider. Round-robin only rotates keys WITHIN the active provider.
- To switch provider, set `judge.provider` to `deepseek` or `mistral`.

Black-box subject models for the expressed-bias study (config `api_predict_only`):

| short | provider | served model id | keys |
|-------|----------|-----------------|------|
| `deepseek-chat` | DeepSeek | `deepseek-chat` | `DEEPSEEK_API_KEY_1/2` |
| `llama-3.3-70b` | OpenRouter | `meta-llama/llama-3.3-70b-instruct` | `OPENROUTER_API_KEY_1/2` |
| `gpt-oss-20b` | OpenRouter | `openai/gpt-oss-20b` | `OPENROUTER_API_KEY_1/2` |

Step 04 pings every judge key and every subject model id once, reporting status by index
only (no key value is ever printed).

---

## 5. Running

```bash
# everything in order, resume-capable
python run_all.py

# split by hardware
python run_all.py --cpu-only     # 01,02,03,06,10
python run_all.py --gpu-only     # 01,02,03,05,08 (+ prereqs)
python run_all.py --api-only     # 01,02,03,04,07,09

# resume from a step, run a subset, or recompute
python run_all.py --from 05
python run_all.py --only 08 09
python run_all.py --no-resume
```

Each step is also runnable on its own, e.g. `python step_08_rag_local.py`. Steps write rows
incrementally and skip already-done keys, so a crash loses nothing.

**Re-run safety.** Step 01 runs on every invocation and checks both datasets for duplicate
rows, duplicate `(index, language, bias_type)` keys, empty/corrupted fields, missing mask
slots, and broken cross-lingual parallelism; it exits non-zero on hard corruption so the
pipeline stops before producing bad results.

**Leakage gate.** Immediately after the KG is built (step 03), `run_all.py` runs
`tests/test_leakage.py` against the freshly built graph and aborts the whole run if it fails.
The gate proves three things on the real artifacts: (1) no TEST or VAL group is a node in the
graph; (2) the KG is `built_from = train_concepts_only` with test concepts held out; (3) for
every evaluated TEST item, retrieval never returns the item's own stereo/anti surface or its
own concepts. Data leakage therefore cannot silently reappear on a future edit.

Suggested order on a fresh GPU VM:

```bash
pip install -r requirements.txt
python install_flash_attention.py
python step_04_dry_run.py          # confirm keys + model ids first
python run_all.py
```

---

## 6. Outputs (in results/)

| File | From | Meaning |
|------|------|---------|
| `integrity_report.json` | 01 | per-dataset dedup/corruption report |
| `concept_split.json` | 02 | concept -> train/val/test assignment |
| `kg/{nodes.csv, edges.csv, graph.json, kg_stats.json}` | 03 | the train-only MS-SKG |
| `dry_run_report.json` | 04 | provider/key/model + subject-model + flash-attn status |
| `cut_units.csv` | 05 | per model/dataset/language/seed cut signature |
| `e5_heldout.csv`, `e5_summary.json` | 06 | held-out bridge ratio vs random-relabel null |
| `calibration.csv`, `calibration.json` | 07 | retrieval n_facts sweep on VAL; recommended value |
| `rag_local.csv` | 08 | expressed bias per condition, local decoders, TEST concepts |
| `rag_api.csv` | 09 | expressed bias per condition, API subjects, TEST concepts |
| `summary.json`, `forest_plot.csv` | 10 | cut-overlap stats, retrieval improvements with CIs, heterogeneity |

`summary.json` is the headline: multi-seed cross-lingual Jaccard with its random-mask null,
the held-out bridge lift, and per-condition retrieval improvement with bootstrap CIs and
Cohen's d (kg_rag vs base, safety_prompt, translate_en, flat_dict, kg_rag_monolingual).

---

## 7. Tests

```bash
python tests/test_no_secrets.py   # fails if any credential is hard-coded
python tests/test_core.py         # split determinism, retrieval self-exclusion, stats
python tests/test_leakage.py      # proves no leakage on the real built graph (run after step 03)
```

All three run under `pytest`. `test_leakage.py` is also enforced automatically as a hard gate
inside `run_all.py` after step 03.

---

## 8. How the outputs answer the review

- **"Graph built from the test set."** Fixed: `kg_stats.json` reports `built_from:
  train_concepts_only` with `n_test_concepts` held out; retrieval runs on TEST only.
- **"Retrieval leaks the gold answer."** Fixed: `rag_leakfree` excludes the query's own
  surfaces and concept; conditions include `flat_dict` and `kg_rag_monolingual` ablations
  plus `safety_prompt` and `translate_en` baselines.
- **"Single-seed disjointness."** Fixed: `cut_units.csv` over 5 seeds; `summary.json` reports
  mean/std cross-lingual Jaccard against a matched-sparsity random-mask null and within-
  language seed stability.
- **"0.767 is circular."** Replaced by held-out `same_as` link prediction vs a random-relabel
  null (`e5_summary.json`).
- **"No baselines."** Added: generic safety prompt, translate-to-English, flat dictionary,
  and a monolingual (no cross-lingual `same_as`) ablation.
- **"Over-correction."** Calibrated: `step_07` picks `n_facts` on VAL concepts to land near
  neutral; TEST runs use the calibrated value.
