# PAPER 6 — BUILD PROMPT (Knowledge-Graph-Bridged Debiasing)

> **This file is a self-contained mega-prompt for an AI coding agent.** Build the
> Paper 6 pipeline exactly as specified. Paper 6 extends Papers 1–5 with a
> **graph-based solution** for the Elsevier special issue *Graph-Based Solutions for
> Artificial Intelligence*. The single most important rule: **reuse the results and
> code of Papers 1–5 wherever a baseline already exists — never burn GPU hours
> recomputing something that is already saved on disk.**

---

## 0. Role and objective

You are an ML research engineer. Build `Codes/paper6/` following the exact
conventions of `Codes/paper5/`. The scientific goal:

Papers 1–5 showed that per-language bias circuits are near-disjoint (mean
Jaccard ≈ 0.035), so English-built debiasing does not transfer to Hindi/Bengali.
Paper 6 supplies the missing cross-lingual structure **externally** with a
**Multilingual Stereotype Knowledge Graph (MS-SKG)** and uses graph algorithms over
it to (E2) augment, (E3) bridge-transfer, and (E4) prompt-debias — while reusing
Papers 1–5 as baselines.

Implement five experiments: **E1** (build the MS-SKG), **E2** (KG-guided
counterfactual augmentation), **E3** (KG-bridged cross-lingual transfer — the
headline), **E4** (KG retrieval-augmented inference), **E5** (graph-propagation
analysis).

---

## 1. Hard constraints

1. **Reuse first (GPU budget).** Before computing anything, check the reuse map in
   §2. If a number exists in a Paper 1–5 `results/*.csv`, **load it; do not recompute.**
2. **Conformity.** Identical models, datasets, quantisation, and metric definitions
   as Papers 1–5 (§3). New numbers must be directly comparable to the old ones.
3. **Single entry point.** `python run_all.py` runs everything. Support
   `--experiment {e1,e2,e3,e4,e5}` and `--resume`.
4. **Resume-capable checkpointing.** Incremental CSV/JSON writes after every unit of
   work; on restart, skip rows already present. Never lose progress on a crash.
5. **Secrets only in `.env`.** Provide `.env.example` (empty values). Round-robin any
   API keys. No key is ever hard-coded, logged, or written to results.
6. **Robust JSON extraction** for any LLM output (reuse `json_extract.py`).
7. **Citation comments.** Top of each method file: a `# Implements ... [Author, year]`
   comment for the method it realises (CDA, GraphRAG, R-GCN, PageRank, etc.).
8. **No fabricated numbers.** Every CSV cell comes from a real computation or a
   reused Paper 1–5 file. If an experiment cannot run, write the row as `NaN` with a
   `status` column, never a made-up value.

---

## 2. The reuse map — load these, do NOT recompute

All paths relative to `Codes/`. Wire these into `config/default.yaml` under `paths`.

| Need | Reuse this file | Use as |
|---|---|---|
| Parallel bias data (EN/HI/BN, `Index`-aligned) | `paper1/data/raw/multicrows_pairs.csv`, `paper1/data/raw/indian_multilingual_bias.csv` | source for the KG **and** all pairs |
| Competence scores (gate cross-lingual numbers) | `paper4/results/competence_by_language.csv` | competence gate (chance 0.25, adequate ≥0.40) |
| Emergence band L\* per model/dataset | `paper4/results/emergence_layers.csv`; `paper5/results/minimal_cut.csv` (`emergence_band`, `total_units`) | **reuse the band — do not re-localise globally** |
| Logit-lens trajectories | `paper4/results/layer_bias_trajectory.csv` | per-concept band refinement input |
| Working **per-language** repair (upper bound) | `paper4/results/filter_results.csv` (`filter_bias_reduction` per language); `paper5/results/cut_results.csv` (`learned_min_cut`) | E2/E3 upper-bound + no-CDA baseline |
| **Failed direct transfer** (the E3 baseline) | `paper5/results/cut_results.csv` (cross-language rows) + `paper5/results/circuit_overlap.csv` (Jaccard ≈ 0.035) | E3 baseline + post-bridge alignment comparison |
| Cut sizes / total units | `paper5/results/minimal_cut.csv` | match cut sizes; reuse `total_units` |
| **Expressed-bias baseline** for prompts | `paper3/results/llm_clti.csv`, `paper3/results/expressed_bias_long.csv`, `paper3/results/suppression_stg.csv` | E4 baseline (plain English debias prompt) |
| Projection-debiasing transfer | `paper2/results/transfer_results.csv` | secondary mitigation baseline |

**Code to import/copy from `paper5/src/` (do not reimplement):**
- `backbone_llm.py` -> `LocalCausalLLM` (load, `sequence_log_likelihood`, `hidden_states`,
  `logit_lens_gap`, `insert_hook`, `generate`).
- `backbone.py` -> encoder loading.
- `data_loader.py` -> `load_bias_dataset`, `_parse_target`, `_fill`, `load_hellaswag_subset`.
- `localize.py` -> `band_from_lens`, `llm_logit_lens_trajectory`.
- `minimal_cut.py` -> `ResidualKeepMask`, `learn_minimal_cut`.
- `circuit_xling.py` -> `jaccard`, `overlap_matrix`.
- `metrics.py` -> `circuit_size_fraction`, `circuit_clti`, `competence_gated_circuit_clti`,
  `beats_controls`.
- `env_utils.py`, `json_extract.py`, `judge_client.py`, `prompts.py`, `reporting.py`.

Prefer `from paper5.src import ...` via a path shim; copy only if import is impractical.

**GPU is spent ONLY on:** E2 KG-CDA repair training, E3 per-concept re-localisation +
cut on the **two local decoders** (band reused, search restricted), E4 KG-RAG on the
two local decoders. E1 and E5 are CPU-only. Everything else is loaded from §2.

---

## 3. Models, datasets, quantisation (must match Papers 1–5)

```yaml
white_box:
  encoders:
    - {hf_id: xlm-roberta-base,              short: xlm-r-base}
    - {hf_id: google/muril-base-cased,       short: muril-base}
    - {hf_id: bert-base-multilingual-cased,  short: mbert}
  llms:
    - {hf_id: Qwen/Qwen2.5-7B-Instruct,       short: qwen2.5-7b}
    - {hf_id: CohereForAI/aya-23-8B,          short: aya-23-8b}
api_predict_only: [gpt-oss-20b, deepseek-chat, llama-3.3-70b]   # E4 only, via API
quantization: {load_in_4bit: true, bnb_4bit_quant_type: nf4,
               bnb_4bit_compute_dtype: bfloat16, bnb_4bit_use_double_quant: true}
datasets:
  crows_pairs:  {file: multicrows_pairs.csv,        languages: [en, hi, bn]}
  indian_bias:  {file: indian_multilingual_bias.csv, languages: [en, hi, bn]}
  hellaswag:    {file: hellaswag_multilingual_val.jsonl, competence_subset_per_language: 500}
```

**Dataset facts the agent must exploit:** both bias CSVs share columns `Index,
Target_Stereotypical, Target_Anti-Stereotypical, Sentence, (stereo_antistereo),
bias_type, language`. **Rows with the same `Index` are translations of one another
across `en/hi/bn`.** This parallelism is the backbone of the KG (free cross-lingual
links) and of the E3 concept bridge. `bias_type` ∈ {gender, religion, caste, race,
nationality, …}. A pair = `_fill(Sentence, Target_Stereotypical)` vs
`_fill(Sentence, Target_Anti-Stereotypical)`.

---

## 4. Directory layout and engineering

```
Codes/paper6/
  README.md
  EXPERIMENT_PLAN.md
  requirements.txt
  .env.example
  run_all.py                 # single entry point; --experiment, --resume
  config/default.yaml
  src/
    __init__.py
    kg_build.py              # E1
    kg_io.py                 # load/save MS-SKG (nodes.csv, edges.csv, graph.json)
    kg_algos.py              # PageRank, label propagation, community detection, R-GCN(optional)
    cda.py                   # E2 counterfactual augmentation from the KG
    bridge_transfer.py       # E3 KG-bridged re-localise + cut
    kg_rag.py                # E4 retrieval-augmented inference
    propagation_analysis.py  # E5
    reuse.py                 # loaders for Paper 1-5 result CSVs (the reuse map)
    backbone_llm.py          # import/copy from paper5
    ... (other reused modules per §2)
  results/                   # all outputs (CSV/JSON), incremental
  tests/                     # unit tests (KG schema, jaccard, BTG, parsers)
```

Engineering: incremental append with a unique key per row (skip if present); a
`status` column on every results CSV (`ok|skipped|nan`); `--resume` re-scans results
and continues; deterministic seeds `[42]`; log to `run_all.log` (no secrets).

---

## 5. E1 — Build the Multilingual Stereotype Knowledge Graph (MS-SKG)  *(CPU)*

**Implements:** stereotype KG from parallel diagnostic data + ConceptNet/Wikidata
extension.

1. **Nodes.**
   - `group` nodes from `Target_Stereotypical`/`Target_Anti-Stereotypical` where the
     target denotes a social group; `attribute` nodes for stereotype descriptors.
     Use `_parse_target` to normalise. One node per (surface_form, language).
   - `canonical_id`: collapse the three language surfaces of the **same `Index`
     target** into one canonical concept (this is the cross-lingual anchor).
2. **Edges.**
   - `stereotype_of` (group->attribute) and `anti_stereotype_of` (group->attribute),
     mined per row; weight = co-occurrence count.
   - `same_as` (cross-lingual, EN<->HI<->BN) — **derived directly from `Index` alignment**
     (free, no API). Optionally extend with Wikidata/ConceptNet `sameAs` for
     out-of-data entities.
   - `is_a` / `related_to` from ConceptNet (optional; flag `use_conceptnet`).
3. **Output** `results/kg/`: `nodes.csv` (id, lang, surface, type, canonical_id,
   bias_type), `edges.csv` (src, dst, relation, weight), `graph.json` (NetworkX
   node-link), `kg_stats.json` (counts, coverage).
4. **Metric — KG coverage:** fraction of dataset stereotype pairs whose group and
   attribute both appear as KG nodes/edges. Target ≥ 0.95 (it is built from the data).

---

## 6. E2 — KG-guided counterfactual data augmentation (CDA)  *(GPU: repair only)*

**Implements:** Counterfactual Data Augmentation for debiasing, KG-guided.

1. For each language, for each `stereotype_of(group->attr)` edge, generate
   counterfactuals by substituting `group` with KG-sibling groups (same `bias_type`
   cluster, via community detection in `kg_algos.py`). Use `_fill` to realise
   sentences. This widens stereotype coverage with **valid, multilingual** swaps.
2. Retrain the **Paper 4/5 repair** (`learn_minimal_cut`, and the encoder graph
   filter `train_filter`) on the KG-augmented pair set, per language, per model.
3. **Baselines (load, do not recompute):** no-CDA = `paper4/results/filter_results.csv`
   and `paper5/results/cut_results.csv`. **Recompute only:** random-swap CDA (no KG)
   and KG-CDA.
4. **Output** `results/e2_cda.csv`: subject_model, dataset, language, method
   {no_cda|random_cda|kg_cda}, bias_baseline, bias_after, bias_reduction,
   stereotypes_covered, n_pairs, seed, status.
5. **Hypothesis to test:** `kg_cda ≥ random_cda ≥ no_cda` on bias_reduction and on
   stereotypes_covered.

---

## 7. E3 — KG-bridged cross-lingual repair transfer  *(headline; GPU: 2 decoders)*

**Implements:** concept-level transfer via the MS-SKG `same_as` bridge.

Scope: **two local decoders** (qwen2.5-7b, aya-23-8b) × two datasets. **Reuse the
emergence band** from `paper4/results/emergence_layers.csv` / `paper5/minimal_cut.csv`
— do not re-localise globally.

Three conditions per target language ℓ ∈ {hi, bn}, grouped by KG concept cluster
(per `bias_type`):
- **(A) Direct transfer (baseline, expected to fail).** Take the English `learned_min_cut`
  unit set; apply those *same* unit indices as a cut on ℓ. If the English unit indices
  are not saved as an artifact, recompute the English cut **once** with
  `learn_minimal_cut(llm, pairs_en, band, cfg)` and cache the mask. Cross-check against
  `paper5/results/circuit_overlap.csv`.
- **(B) Per-language upper bound (load).** `paper5/results/cut_results.csv`
  `learned_min_cut` for ℓ.
- **(C) KG-bridged (proposed).** For each KG concept cluster: follow `same_as` to the
  ℓ-language pairs of the **same `Index` set**, refine the band per language with
  `band_from_lens` on those pairs, learn a cut with `learn_minimal_cut`, evaluate FBR.
  Aggregate over clusters (size-matched to A via `circuit_size_fraction`).
- **Post-bridge alignment:** recompute `jaccard` between the KG-aligned per-language
  cut sets; compare to the 0.035 baseline.

**Output** `results/e3_bridge.csv`: subject_model, dataset, target_lang, bias_type,
condition {direct|upper_bound|kg_bridged}, cut_size, bias_baseline, bias_after,
fbr, n_pairs, seed, status. Plus `results/e3_btg.csv` with **Bridge Transfer Gain**
`BTG = fbr(kg_bridged) − fbr(direct)` per (model, dataset, target_lang).

**Success:** `fbr(kg_bridged) > fbr(direct) ≈ 0`, approaching `upper_bound`; `BTG > 0`.

---

## 8. E4 — KG retrieval-augmented inference (GraphRAG)  *(API + 2 local decoders)*

**Implements:** retrieval-augmented prompting with a knowledge graph (GraphRAG-style).

1. For each test pair, extract its group/attribute, retrieve `anti_stereotype_of` and
   `related_to` neighbours from the MS-SKG **in the test language**, and inject them as
   a short counter-stereotypical context before the question.
2. Models: qwen2.5-7b, aya-23-8b (local) + gpt-oss-20b, deepseek-chat, llama-3.3-70b
   (API, round-robin keys). Measure **expressed** bias (reuse `prompts.py`,
   `judge_client.py`, `json_extract.py`, refusal handling).
3. **Baseline (load):** plain English debias prompt = `paper3/results/llm_clti.csv`.
   **Recompute only:** the KG-RAG condition.
4. **Output** `results/e4_kgrag.csv`: subject_model, access {local|api}, dataset,
   language, method {plain_prompt|kg_rag}, expressed_bias, drop_vs_nocontext,
   refusals, n, status.
5. **Hypothesis:** KG-RAG lowers expressed bias more consistently across languages
   than the plain English prompt; strongest on the larger API models.

---

## 9. E5 — Graph-propagation analysis  *(CPU)*

**Implements:** Personalised PageRank / label propagation (and optional R-GCN) over
the MS-SKG.

1. Seed propagation from groups with high measured intrinsic bias (load intrinsic
   scores from `paper3/results/intrinsic_bias_long.csv` and the bias scores implied by
   Papers 4/5). Compute a graph-derived stereotype-strength score per node.
2. Correlate graph score with measured per-model bias; report Pearson r and a
   held-out predictive check.
3. **Output** `results/e5_propagation.csv`: subject_model, node_canonical_id,
   bias_type, graph_score, measured_bias, and a top-level `results/e5_correlation.json`
   (r, n, p).

---

## 10. Output CSV schemas (summary)

`results/kg/nodes.csv`, `results/kg/edges.csv`, `results/kg/kg_stats.json`,
`results/e2_cda.csv`, `results/e3_bridge.csv`, `results/e3_btg.csv`,
`results/e4_kgrag.csv`, `results/e5_propagation.csv`, `results/e5_correlation.json`,
`results/metrics_summary.csv` (one tidy roll-up), `results/integrity_report.json`
(reuse-vs-recompute audit: which baselines were loaded, which rows were computed).

Every results CSV carries `status` and, where applicable, `seed` and `n_pairs`.

---

## 11. `run_all.py` orchestration and run order

Order (each step independently resumable and reportable):
1. **E1** build MS-SKG (CPU; locks in the artifact).
2. **E3** on crows_pairs first, both decoders (headline; reuses band + baselines).
3. **E2** KG-CDA (encoders + decoders).
4. **E4** KG-RAG (API first — cheap, no local GPU; then 2 local decoders).
5. **E5** propagation analysis (CPU).

`run_all.py` prints, at the end, the reuse audit (GPU-hours-equivalent saved by
loading Paper 1–5 baselines) and a one-line pass/fail per experiment.

---

## 12. Tests and acceptance criteria

`tests/` must cover: KG schema validity (no dangling edges; every `same_as` links a
shared `canonical_id`); `jaccard` and `BTG` correctness on toy inputs; counterfactual
generator preserves the masked slot; CSV resume (no duplicate rows); secret-safety
(no key in any results/log file).

**Definition of done:** `python run_all.py` produces all §10 outputs;
`integrity_report.json` shows the §2 baselines were **loaded, not recomputed**;
E3 yields `BTG > 0` for at least one (model, dataset, language); E1 coverage ≥ 0.95;
no secret appears in any artifact.

---

## 13. Citations to attach in code comments

CrowS-Pairs [Nangia et al., 2020]; SentenceDebias [Liang et al., 2020]; INLP
[Ravfogel et al., 2020]; logit lens [Geva et al., 2022]; L0 / movement pruning
[Louizos et al., 2018; Sanh et al., 2020]; GraphGPS [Rampášek et al., 2022]; R-GCN
[Schlichtkrull et al., 2018]; personalised PageRank [Haveliwala, 2002]; counterfactual
data augmentation [e.g., Zmigrod et al., 2019]; ConceptNet [Speer et al., 2017];
Wikidata [Vrandečić & Krötzsch, 2014]. (Verify exact references against
`Submission/references.bib` before writing the paper.)
```
