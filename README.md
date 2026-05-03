# doubleword-deepseek-eval

Compare **DeepSeek-V4-Flash** (via Doubleword API) against **gemini-3-flash-preview** (Google Gemini API) on two medical AI tasks, evaluated by **gemini-3.1-pro-preview** as judge.

---

## What it tests

### Part 1 — Long Context (Needle-in-a-Haystack)
- Streams clinical notes from HuggingFace (`AGBonnet/augmented-clinical-notes`)
- **Four context tiers**: 200k / 400k / 600k / 800k tokens — each tier uses a different random seed so background notes differ, but both models see the same notes at each tier
- Truncates to each model's actual context limit (`DEEPSEEK_MAX_CONTEXT`, `GEMINI_MAX_CONTEXT`)
- Plants **5 specific clinical facts** at depths 5 / 25 / 50 / 75 / 95 % through each corpus
- Both models answer 5 QA questions per tier (20 total); Gemini Pro scores accuracy and groundedness (1–5 each)
- Corpora cached to `results/corpus_cache_{tier}k.txt` — delete to force fresh download

### Part 2 — Tool Calling
Five ICU/emergency clinical scenarios:

| # | Scenario | Key Tools |
|---|----------|-----------|
| 1 | Septic shock | qSOFA · SOFA · GCS |
| 2 | Post-cardiac surgery ICU | APACHE II · SOFA |
| 3 | DVT risk stratification | Wells DVT · BMI |
| 4 | Traumatic brain injury | GCS · BMI |
| 5 | AECOPD / hypercapnic respiratory failure | BMI · convert_units · drug_interactions · guidelines |

Gemini Pro scores each scenario on tool selection (0–3), parameter accuracy (0–3), and clinical reasoning (0–4).

---

## Setup

```bash
git clone https://github.com/dcstang/doubleword-deepseek-eval
cd doubleword-deepseek-eval
pip install -r requirements.txt
cp .env.example .env   # fill in API keys
```

### Required environment variables

| Variable | Description |
|----------|-------------|
| `DOUBLEWORD_API_KEY` | Doubleword API key (OpenAI-compatible) |
| `GEMINI_API_KEY` | Google Gemini API key |

See `.env.example` for all options including model names, context limits, and pricing.

---

## Running

```bash
# Full evaluation (Part 1 + Part 2 + Judge + Cost report)
python main.py

# Skip long-context entirely (re-use saved results, re-run judge)
python main.py --skip-long-context

# Re-run only Gemini LC (e.g. after fixing a model name error); load saved DeepSeek answers
python main.py --skip-deepseek-lc --skip-tool-calling --skip-judge

# Re-run only DeepSeek LC; load saved Gemini answers
python main.py --skip-gemini-lc --skip-tool-calling --skip-judge

# Skip both model evals, only re-run judge on saved results
python main.py --skip-long-context --skip-tool-calling

# Skip judge (generate cost/report from saved eval)
python main.py --skip-long-context --skip-tool-calling --skip-judge
```

---

## Batch / Delayed mode

Doubleword's **delayed API** (`completion_window="1h"`) is enabled by default for long-context queries (`USE_DOUBLEWORD_BATCH=true`). This uses the OpenAI Batch API to submit all **20** QA questions (4 tiers × 5 needles) as a single job, reducing cost by ~50%.

Gemini batch (`USE_GEMINI_BATCH=true`) attempts the `google-genai` batch API and falls back to concurrent async requests.

---

## Output

All results are written to `results/`:

| File | Contents |
|------|---------|
| `long_context_results.json` | Raw model answers — 4 tiers × 5 QA questions |
| `tool_calling_results.json` | Tool calls and responses for 5 scenarios |
| `long_context_evaluation.json` | Gemini Pro scores per tier for Part 1 |
| `tool_calling_evaluation.json` | Gemini Pro scores per scenario for Part 2 |
| `cost_summary.json` | Token usage and estimated USD costs (standard vs batch) |
| `final_report.md` | Human-readable comparison table and verdict |
| `corpus_cache_{200k,400k,600k,800k}.txt` | Cached corpora (delete to force re-download) |

---

## Cost comparison

The report includes a side-by-side cost table:

| Model | Standard | Batch/Delayed | Savings |
|-------|:--------:|:-------------:|:-------:|
| DeepSeek-V4-Flash (Doubleword) | calculated | calculated | ~50% on long-ctx |
| Gemini 3 Flash | calculated | calculated | ~50% if batch |
| Gemini 3 Pro (judge) | estimated | n/a | — |

Pricing defaults are in `.env.example` — verify current rates at [doubleword.ai](https://doubleword.ai) and [ai.google.dev/pricing](https://ai.google.dev/pricing).
