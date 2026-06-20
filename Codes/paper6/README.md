# Paper 6 - Knowledge-Graph-Bridged Multilingual Debiasing (MS-SKG)

One-stop, resume-capable pipeline for the Elsevier *Graph-Based Solutions for AI*
special issue. It builds a Multilingual Stereotype Knowledge Graph (MS-SKG) and uses
it to bridge the near-disjoint, language-specific bias circuits that Papers 1-5
documented (cross-lingual circuit Jaccard ~= 0.035). Everything reuses the verified
Paper 5 compute (`LocalCausalLLM`, `learn_minimal_cut`, emergence band) and the
Papers 2-5 result CSVs as baselines, so almost no GPU is re-spent.

This README is self-contained: read it top to bottom and you can run the whole study.

---

## 1. What it does (five experiments)

| Exp | Question | New compute | Reused baseline |
|-----|----------|-------------|-----------------|
| E1 | Build the MS-SKG from the parallel diagnostic data | CPU only | - |
| E2 | Does KG-guided counterfactual augmentation improve the repair? | 2 cut trainings / decoder | Paper 5 no-CDA cut |
| E3 | **Headline.** Does the KG bridge a repair across languages? | 1 English cut / (decoder, dataset) | Paper 5 per-language cut + baselines |
| E4 | Does KG-retrieval-augmented inference cut *expressed* bias? | short generations / API calls | Paper 3 `llm_clti` (reference) |
| E5 | Do graph paths bridge what parameters do not? | CPU only | Paper 5 `circuit_overlap` (Jaccard) |

Headline metrics: **Bridge Transfer Gain** (E3) `BTG = FBR(kg_bridged) - FBR(direct)`,
and the **bridge ratio** (E5), the share of cross-lingual PageRank mass landing on
true `same_as` counterparts.

---

## 2. Folder layout and the execution sequence

**The execution order is the numeric SUFFIX on each folder name.** Sort the folders
and run them in that order: `Dataset_Prep_01`, `Dataset_Prep_02`, `CPU_Only_03`,
`GPU_Only_04`, ... Each numbered folder holds exactly one runnable script.
`Common_00` (suffix `00`) is the shared library imported by every step - it is not an
execution step, and its `00` puts it first in any sorted listing.

```
Codes/paper6/
  run_all.py                      <- single entry point (runs 01..08 in order)
  config/default.yaml             <- all paths, models, hyperparameters, provider config
  .env.example                    <- copy to .env and fill in keys (never committed)
  requirements.txt                <- global install (no venv)

  Common_00/                      SHARED LIBRARY (imported by all steps; not executed)
    common.py                     config/env, round-robin keys, CSV resume, parsers
    providers.py                  judge client (gemini primary, no fallback)
    api_subjects.py               E4 API subject-model routing
    reuse.py                      loaders for Papers 2-5 baselines
    kg_algos.py                   siblings, counter-stereotype, bridge, PageRank
    e4_core.py                    E4 RAG context + expressed-bias evaluation
    dataio.py                     read + normalise the parallel bias CSVs
    kg_io.py                      MS-SKG save/load (nodes, edges, graph.json)
    model_utils.py                (GPU) reuse Paper 5 LocalCausalLLM / minimal cut
    install_flash_attention.py    (GPU setup) download a PRE-COMPILED flash-attn wheel

  Dataset_Prep_01/check_data.py   STEP 01  integrity + dedup + corruption check     [CPU]
  Dataset_Prep_02/build_kg.py     STEP 02  E1: build the MS-SKG                      [CPU]
  CPU_Only_03/dry_run.py          STEP 03  test every provider key + model id        [CPU]
  GPU_Only_04/e3_bridge.py        STEP 04  E3 KG-bridged transfer (headline)         [GPU]
  GPU_Only_05/e2_cda.py           STEP 05  E2 KG-guided counterfactual augmentation  [GPU]
  CPU_Only_06/e4_kgrag_api.py     STEP 06  E4 KG-RAG, API subject models             [CPU/API]
  GPU_Only_07/e4_kgrag_local.py   STEP 07  E4 KG-RAG, local decoders                 [GPU]
  CPU_Only_08/e5_propagation.py   STEP 08  E5 graph propagation analysis             [CPU]

  tests/
    test_core.py                  parsers, CSV resume, KG schema/algorithms, BTG
    test_no_secrets.py            guard: no hard-coded credentials anywhere
```

**Execution order**

| Order | Folder / script | Hardware | Experiment |
|------:|-----------------|----------|------------|
| 00 | `Common_00/` (shared library) | - | imported, not run |
| 01 | `Dataset_Prep_01/check_data.py` | CPU | prerequisite |
| 02 | `Dataset_Prep_02/build_kg.py` | CPU | E1 |
| 03 | `CPU_Only_03/dry_run.py` | CPU | prerequisite (API test) |
| 04 | `GPU_Only_04/e3_bridge.py` | GPU | E3 (headline) |
| 05 | `GPU_Only_05/e2_cda.py` | GPU | E2 |
| 06 | `CPU_Only_06/e4_kgrag_api.py` | CPU + API | E4 (API subjects) |
| 07 | `GPU_Only_07/e4_kgrag_local.py` | GPU | E4 (local decoders) |
| 08 | `CPU_Only_08/e5_propagation.py` | CPU | E5 |

The folder name tells you both the order (suffix) and the hardware (`CPU_Only` /
`GPU_Only` / `Dataset_Prep`). `run_all.py` builds the plan automatically: steps 01-02
(data + KG) always precede any experiment, step 03 (dry run) precedes only the
API-using steps (06, 07).

---

## 3. Setup (global environment, no venv)

Install into the global interpreter exactly as requested - do not create a venv.

```bash
cd Codes/paper6
pip install -r requirements.txt
```

Flash-attention is installed separately, from a **pre-compiled** wheel (never built
from source), and only where it is supported (Linux x86_64 + CUDA). Run this on the
GPU machine before the GPU steps:

```bash
python Common_00/install_flash_attention.py            # auto-detect torch/CUDA/python/ABI
python Common_00/install_flash_attention.py --version 2.8.3
```

The script detects torch version, CUDA, Python tag, and C++11 ABI, constructs the
matching `Dao-AILab/flash-attention` release wheel URL, verifies it exists, then
`pip install`s it with `--no-build-isolation`. On non-Linux / non-CUDA hosts it
skips cleanly and the decoders fall back to PyTorch SDPA attention (slower, still
correct). `config.backbone.use_flash_attention: true` makes `LocalCausalLLM` request
flash-attention when the wheel is present.

---

## 4. Secrets and the judge (.env)

Copy the template and fill in real values. The real `.env` is git-ignored and is the
**only** place a key may live; no key value appears in any tracked file (enforced by
`tests/test_no_secrets.py`).

```bash
cp .env.example .env       # then edit .env
```

Judge / answer-extraction provider (config `judge`):

- **PRIMARY: Gemini 2.5 Flash**, four keys, round-robin. Provide them as either
  `GCP_Key1..GCP_key4` **or** `GEMINI_API_KEY_1..4` - the config lists both names and
  uses whichever are present (Google AI Studio keys work on the OpenAI-compatible
  endpoint). Thinking is disabled via `reasoning_effort: none` so a small token
  budget still returns visible text.
- **DeepSeek**, **Mistral Small**, **OpenRouter** are selectable alternatives
  (`judge.provider`). Two keys each, round-robin.
- **No automatic cross-provider fallback.** The active provider is fixed by
  `judge.provider`; a failed call is recorded as a failure, never silently retried on
  another provider. Round-robin only rotates keys *within* the active provider.

E4 black-box subject models (config `api_predict_only`) route through
`Common_00/api_subjects.py`:

| short | provider | served model id | keys |
|-------|----------|-----------------|------|
| `deepseek-chat` | DeepSeek | `deepseek-chat` | `DEEPSEEK_API_KEY_1/2` |
| `llama-3.3-70b` | OpenRouter | `meta-llama/llama-3.3-70b-instruct` | `OPENROUTER_API_KEY_1/2` |
| `gpt-oss-20b` | OpenRouter | `openai/gpt-oss-20b` | `OPENROUTER_API_KEY_1/2` |

Step 03 pings every judge key **and** every subject model id once, reporting status
by index only (no key value is ever printed).

---

## 5. Models and data (identical to Papers 1-5)

Encoders: `xlm-roberta-base`, `google/muril-base-cased`, `bert-base-multilingual-cased`.
Local decoders (NF4 4-bit): `Qwen/Qwen2.5-7B-Instruct`, `CohereForAI/aya-23-8B`.

Datasets (parallel by `Index` across en/hi/bn): `multicrows_pairs.csv`,
`indian_multilingual_bias.csv`, and `hellaswag_multilingual_val.jsonl` for the
competence gate. The cross-lingual parallelism is the backbone of the KG: same-Index
rows in different languages give free `same_as` edges. `indian_bias` stores several
`bias_type` rows under one `Index`, so the true alignment unit is `(Index, bias_type)`
- handled throughout (the integrity check treats this as valid, not duplication).

---

## 6. Reuse map (where the GPU savings come from)

`Common_00/reuse.py` loads these instead of recomputing them; `run_all.py` prints a
reuse audit at start.

| File | Paper | Used for |
|------|-------|----------|
| `paper5/cut_results.csv` | 5 | E3 per-language KG-bridged cut + baselines; E2 no-CDA |
| `paper5/minimal_cut.csv` | 5 | emergence band per (model, dataset) |
| `paper5/circuit_overlap.csv` | 5 | E5 circuit Jaccard contrast |
| `paper4/emergence_layers.csv` | 4 | emergence-band fallback |
| `paper4/competence_by_language.csv` | 4 | competence gate (low-competence flagged) |
| `paper3/llm_clti.csv` | 3 | E4 expressed-bias reference |
| `paper2/transfer_results.csv` | 2 | projection-transfer reference |

---

## 7. Running

```bash
# everything, in order, resume-capable
python run_all.py

# one experiment (prerequisites added automatically)
python run_all.py --experiment e3        # 01,02,04
python run_all.py --experiment e4        # 01,02,03,06,07

# split by hardware
python run_all.py --cpu-only             # 01,02,03,06,08
python run_all.py --gpu-only             # 01,02,03,04,05,07

# resume from a step, or recompute from scratch
python run_all.py --from 04
python run_all.py --no-resume
```

Each step is also runnable on its own, e.g. `python GPU_Only_04/e3_bridge.py`. Every
step writes rows incrementally and skips already-done keys, so a crash loses nothing -
re-running continues where it stopped.

**Re-run safety.** Step 01 runs on every invocation and checks for duplicate rows,
duplicate `(Index, language, bias_type)` keys, empty/corrupted fields, missing mask
slots, and broken cross-lingual parallelism; it exits non-zero on hard corruption so
the pipeline stops before producing bad results.

---

## 8. Outputs (in `results/`)

| File | From | Key columns |
|------|------|-------------|
| `data_integrity_report.json` | 01 | per-dataset dedup/corruption report |
| `kg/{nodes.csv, edges.csv, graph.json, kg_stats.json}` | 02 | the MS-SKG |
| `dry_run_report.json` | 03 | provider/key/model + subject-model status |
| `e3_bridge.csv`, `e3_btg.csv` | 04 | FBR per condition; Bridge Transfer Gain |
| `e2_cda.csv` | 05 | bias reduction for no/random/KG CDA |
| `e4_kgrag_api.csv` | 06 | expressed bias, no_rag vs kg_rag (API subjects) |
| `e4_kgrag_local.csv` | 07 | expressed bias, no_rag vs kg_rag (local decoders) |
| `e5_propagation.csv`, `e5_summary.json` | 08 | bridge ratio, lift, Jaccard contrast |

---

## 9. Tests

```bash
python tests/test_core.py          # parsers, CSV resume, KG schema/algorithms, BTG
python tests/test_no_secrets.py    # fails if any credential is hard-coded
```

Both also run under `pytest`. The no-secrets guard scans every tracked file (code,
configs, results) for credential-shaped strings and for any key env-var assigned a
literal value; only `.env` is exempt.

---

## 10. Build specification

`PAPER6_BUILD_PROMPT.md` is the full mega-prompt this implementation follows (reuse
map, model/data list, per-experiment design, CSV schemas, run order).
