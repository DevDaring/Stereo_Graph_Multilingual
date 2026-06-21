# A Graph-Based Account of Multilingual Stereotype Bias

Code, knowledge graph, and result files for the paper
*A Graph-Based Account of Multilingual Stereotype Bias: From Detection to a
Knowledge-Graph Bridge for Cross-Lingual Repair.*

This repository accompanies a study of stereotype bias across **English, Hindi, and
Bengali** on one shared set of models, from frozen encoders to instruction-tuned
decoders. Six linked studies run on the same data and the same bias metric. The first
five build a diagnosis. The sixth acts on it with an external knowledge graph.

Anonymous mirror: <https://anonymous.4open.science/r/Stereo_Graph_Multilingual>

---

## The one-paragraph story

A fairness fix is usually built and checked in English. It rarely holds when the input
is Hindi or Bengali. Studies 1 to 5 show why: the per-language bias circuits overlap by
a mean Jaccard index of about **0.035**, so an English-fitted repair finds little shared
structure to act on in another language. Study 6 supplies that missing structure from
outside the model. A multilingual stereotype knowledge graph aligns the same social
group across the three languages, and graph propagation carries a mean of **0.767** of
its cross-lingual mass onto same-concept counterparts. A graph-bridged repair then gains
a mean of **2.59** points on the Indian-bias benchmark, where in-parameter projection and
circuit cuts do not transfer.

---

## The six studies

| # | Folder | What it does | Headline result |
|---|--------|--------------|-----------------|
| 1 | `Codes/paper1` | Detect bias with lightweight heads over a frozen encoder | Typed graph head loses (0.382 vs 0.410); an honest negative result |
| 2 | `Codes/paper2` | Mitigate with SentenceDebias / INLP and test transfer | English-fitted fixes do not transfer (CLTI negative in 11/12) |
| 3 | `Codes/paper3` | Audit five large models: intrinsic vs expressed bias | Safety holds in English, leaks in Hindi/Bengali (STG up to 7.79) |
| 4 | `Codes/paper4` | Locate the emergence layer and fit a graph filter there | English bias reduction up to 9.15 points; identity-guarded |
| 5 | `Codes/paper5` | Cut the minimal bias circuit; measure cross-lingual overlap | Per-language circuits near-disjoint, mean Jaccard 0.035 |
| 6 | `Codes/paper6` | Build the knowledge graph; bridge, retrieve, augment | Bridge ratio 0.767; bridge transfer gain +2.59 on Indian-bias |

Every cross-lingual result is read under a per-language reading-competence control
(four-way HellaSwag accuracy), so a Hindi or Bengali finding is never confused with the
model failing to read the language.

---

## Models

- **Frozen encoders:** XLM-R-base, MuRIL-base, mBERT.
- **Local decoders (4-bit NF4):** Qwen2.5-7B-Instruct, Aya-23-8B.
- **API decoders (expressed-bias audit and retrieval):** DeepSeek-V3, Llama-3.3-70B,
  gpt-oss-20b.

## Datasets

- **Multi-CrowS-Pairs** — 1,422 pairs per language, nine bias types; a translation of
  CrowS-Pairs into English, Hindi, and Bengali.
- **Indian-Bias** — 761 pairs per language; caste, religion, gender, and race in the
  Indian context.
- **Multilingual HellaSwag** — reading-competence control, 500 items per language.

The two bias benchmarks are in `Codes/paper1/data/raw/`. Each item is a context with a
masked slot, filled with a stereotypical or a less-stereotypical target.

---

## Repository layout

```
Codes/
  paper1/data/raw/        the two parallel bias benchmarks (CSV)
  paper2/results/         Study 2 projection-transfer result files
  paper3/results/         Study 3 intrinsic-vs-expressed audit result files
  paper4/results/         Study 4 emergence + graph-filter result files
  paper5/results/         Study 5 minimal-cut + circuit-overlap result files
  paper5/src/             shared compute reused by Study 6 (model loading, minimal cut)
  paper6/                 Study 6: the knowledge graph and the runnable pipeline
    Common_00/            shared library (config, providers, graph algorithms)
    Dataset_Prep_01..02/  data integrity check, knowledge-graph build (E1)
    CPU_Only_03/06/08/    dry run, KG-retrieval (API), graph propagation (E5)
    GPU_Only_04/05/07/    bridged transfer (E3), counterfactual augmentation (E2), KG-retrieval (local)
    results/              Study 6 result files and the knowledge graph (results/kg/)
    run_all.py            single entry point
```

The folder suffix on each Study 6 script is its execution order, e.g. `Dataset_Prep_01`
runs before `CPU_Only_03`. `Common_00` is the shared library, imported by every step.

---

## Result files

| File | Study | Content |
|------|-------|---------|
| `paper2/results/transfer_results.csv` | 2 | per-language projection-debiasing transfer |
| `paper3/results/llm_clti.csv`, `suppression_stg.csv` | 3 | expressed-bias transfer and safety gap |
| `paper4/results/filter_results.csv`, `emergence_layers.csv` | 4 | filter reduction and emergence layer |
| `paper5/results/cut_results.csv`, `circuit_overlap.csv` | 5 | minimal cut and Jaccard overlap |
| `paper6/results/e3_btg.csv` | 6 | bridge transfer gain |
| `paper6/results/e4_kgrag_local.csv`, `e4_kgrag_api.csv` | 6 | KG-retrieval expressed bias |
| `paper6/results/e5_propagation.csv`, `e5_summary.json` | 6 | graph propagation and bridge ratio |
| `paper6/results/kg/nodes.csv`, `edges.csv` | 6 | the multilingual stereotype knowledge graph |

The knowledge graph holds 10,141 nodes, 10,887 cross-lingual `same_as` edges, and 3,109
canonical concepts that span the three languages.

---

## Reproducing Study 6

Study 6 is fully runnable. The earlier studies' result files are reused as baselines, so
no re-run of Studies 1 to 5 is needed.

```bash
cd Codes/paper6
pip install -r requirements.txt          # global environment; no venv
cp .env.example .env                      # then fill in keys (see below)
python run_all.py                         # runs the pipeline in order
```

Useful variants:

```bash
python run_all.py --cpu-only              # data check, KG build, KG-retrieval (API), propagation
python run_all.py --gpu-only              # bridged transfer, augmentation, KG-retrieval (local)
python run_all.py --experiment e5         # one experiment, with its prerequisites
```

**Keys.** Secrets load from `.env`, never from code. The pipeline needs a HuggingFace
token for the local decoders and a judge/answer-extraction provider for the
expressed-bias audit. The judge is selectable (Gemini, DeepSeek, Mistral, or OpenRouter)
with round-robin keys and no cross-provider fallback. See `Codes/paper6/.env.example`.

**GPU.** The local-decoder steps (bridged transfer, KG-retrieval-local) need a single
24 GB GPU. A pre-compiled FlashAttention-2 wheel is installed by
`Codes/paper6/Common_00/install_flash_attention.py` on Linux x86_64 with CUDA; other
hosts fall back to scaled-dot-product attention automatically.

**Re-run safety.** Every step is resume-capable and checks for duplicate or corrupted
data on each run. Results are written incrementally, so a crash loses nothing.

---

## Headline findings

- **Bias circuits are language-specific.** Per-language minimal cuts overlap by a mean
  Jaccard index of 0.035; the Hindi-Bengali pair (0.047) overlaps more than the
  English-Indic pairs (0.029).
- **The graph bridges what the parameters do not.** Graph propagation carries 0.767 of
  cross-lingual mass onto same-concept counterparts, against the circuit overlap of 0.035.
- **Graph-bridged transfer recovers fairness gains** on the Indian-bias set (mean +2.59
  points; cleanest competence-valid cell +6.53), where direct transfer of the English cut
  does not (mean -0.45).
- **Graph retrieval steers expressed bias** at inference, on local and API models alike.
  It lands near neutral in English and over-corrects in Hindi and Bengali, scaling with
  the model's prior bias.
- **Honest negatives are kept in.** The typed detection head loses to the untyped graph,
  the bridged-transfer gain is dataset-specific and not statistically separable from zero
  over all eight cells, and the counterfactual augmentation does not help.

All numbers above are read from the result files in this repository.
