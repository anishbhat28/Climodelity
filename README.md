# 🌊 Climodelity

**An ML trust lab for tabular models — surface where your model fails, and why.**

Climodelity takes a trained model and the data it was evaluated on, and runs
a grounded autoresearch loop that proposes, tests, and validates hypotheses
about regimes of systematic failure. Every reported finding is gated by a
held-out split and a Bonferroni-corrected Welch's t-test, and every number
on the dashboard traces back to the exact tool call that produced it.

---

## What the pipeline does

1. **Upload** a training script, a tabular CSV, and a natural-language prompt.
2. **Preprocess** — the script and data features are summarized into a
   `program.md` specification (an instruction set downstream agents can use
   to iterate on the training code).
3. **Train** — your script is executed to produce `predictions.csv`
   (`target`, `prediction`, plus any numeric feature columns).
4. **Autoresearch** — the loop enumerates regime hypotheses over the
   `predictions.csv` fields, tests each on a discovery split, and validates
   the survivors on a held-out split with Bonferroni correction.
5. **Dashboard** — RMSE / MAE, predictions-vs-targets, error and residual
   distributions, and a ranked, receipted panel of validated failure modes.

---

## Setup

Requires Python 3.13 and macOS/Linux. On macOS, `libomp` is needed if your
training script uses XGBoost.

```bash
# clone
git clone https://github.com/anishbhat28/DELULUGE.git climodelity
cd climodelity

# env
python3 -m venv env
source env/bin/activate

# core deps
pip install streamlit pandas numpy scipy matplotlib cmocean \
            scikit-learn google-genai openai joblib werkzeug

# macOS-only, if your train.py uses xgboost
brew install libomp

# model-agnostic extras you may want
pip install xgboost tensorflow torch
```

Set API keys in the shell **before** launching Streamlit so subprocesses
inherit them:

```bash
export GEMINI_API_KEY='your-gemini-key'      # for the autoresearch loop
export OPENAI_API_KEY='your-openai-key'      # for program.md generation
```

Launch the app:

```bash
streamlit run app.py
```

---

## Training-script contract

The uploaded `train.py` is executed with `python train.py` after the data is
saved as `data.csv` in the project root. It must:

1. **Read** `data.csv` (or any hardcoded CSV name — Climodelity auto-aliases
   common names so you rarely need to change your code).
2. **Write** `predictions.csv` in the project root with at minimum the
   columns `target` and `prediction`. Any additional numeric columns are
   picked up automatically as regime-sliceable features.

If your script doesn't write `predictions.csv`, Climodelity auto-injects a
best-effort writer that introspects globals for common names like `y_test` /
`*_preds` and a `test` / `test_df` DataFrame.

---

## Dataset

<!-- Fill in details about your dataset here: source, shape, columns,
     preprocessing notes, licensing, anything a future user should know
     before uploading it. -->

_TODO: dataset description._

---

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│  Streamlit entry  (app.py) — upload page                   │
│    • training script + CSV + prompt                        │
│    • auto-patches hardcoded CSV filenames                  │
│    • injects predictions.csv writer if missing             │
├───────────────────────────────────────────────────────────┤
│  Automated preprocessing  (automated_preprocessing.py)     │
│    • AST-extracts training-loop context from train.py      │
│    • pulls data features from the CSV                      │
│    • emits program.md spec                                 │
├───────────────────────────────────────────────────────────┤
│  Training subprocess                                       │
│    • runs your train.py verbatim                           │
│    • expects predictions.csv on disk                       │
├───────────────────────────────────────────────────────────┤
│  Regime extraction  (rmse_regimes.py)                      │
│    • target / prediction / abs_error / residual /          │
│      residual_sign + feature::<col> for each numeric col   │
├───────────────────────────────────────────────────────────┤
│  Autoresearch loop  (autoresearch.py)                      │
│    • agent proposes regime hypotheses within typed tool    │
│    • 70/30 discovery/validation split (fixed seed)         │
│    • Welch's t-test per hypothesis                         │
│    • Bonferroni correction on validation                   │
│    • writes outputs/findings.json with receipts            │
├───────────────────────────────────────────────────────────┤
│  Dashboard  (pages/dashboard.py)                           │
│    • headline metrics · scatter · error + residual hists   │
│    • validated / rejected failure-mode panels              │
│    • legacy SSH atlas (rendered if the .npz fields exist)  │
└───────────────────────────────────────────────────────────┘
```

---

## File layout

```
climodelity/
├── README.md                       # you are here
├── app.py                          # upload page
├── main.py                         # multipage navigation entry
├── pages/
│   └── dashboard.py                # results dashboard
├── automated_preprocessing.py      # train.py AST + data-feature extraction
├── autoresearch.py                 # hypothesis generation + validation loop
├── rmse_regimes.py                 # generic tabular regime extractor
├── train.py                        # overwritten by uploads at runtime
├── predictions.csv                 # written by train.py (runtime artifact)
├── program.md                      # generated spec (runtime artifact)
└── outputs/
    └── findings.json               # autoresearch execution trace + results
```

---

## Method summary

**Discovery / validation split.** Rows are permuted with a fixed seed and
split 70/30. The agent only sees the discovery split during hypothesis
generation; every surviving candidate is re-tested on the held-out
validation split with Bonferroni correction to control the family-wise
error rate.

**Regime language.** Generic tabular. Always-present fields — `target`,
`prediction`, `abs_error`, `residual`, `residual_sign` — plus one
`feature::<col>` per numeric column in `predictions.csv`. Comparators:
`percentile_gt` / `percentile_lt` (value is 0–100) and `gt` / `lt` / `eq`
(value is a raw threshold).

**Statistical test.** Welch's t-test on absolute error inside vs. outside
the regime. Unequal-variance because percentile-based splits systematically
produce heteroscedastic groups. A discovery result is promoted to a
candidate iff `error_ratio > 1.2` and `p < 0.001`.

**Receipts.** Every finding links to the tool-call id that computed each
number. `outputs/findings.json` contains the full execution trace.

---

## Attribution

The autoresearch loop in this project is directly inspired by **Andrej
Karpathy's autoresearch concept** — the framing of a constrained LLM agent
proposing and testing domain-scoped hypotheses under a budget, with
statistical gatekeeping rather than raw-token introspection. Climodelity's
specific contributions are the typed regime language for tabular ML
failure modes, the train.py contract for closing the loop from raw data to
receipted findings, and the Bonferroni-on-validation gate.

All agent code is written from scratch; no external agent framework is
used.
