"""
Part 1: Long-context needle-in-a-haystack evaluation.

Corpus: numbered clinical notes from AGBonnet/augmented-clinical-notes with 5 needle
notes planted at specific depth positions (5/25/50/75/95%).

Batch flow (Doubleword delayed, 1h window):
  1. submit_long_context_batch()  — upload + create batch job, returns batch_id
  2. <caller runs other work>
  3. collect_long_context_batch() — polls until done, returns ModelResponse list

Gemini runs synchronously (or concurrently via asyncio fallback).
"""

import json
import os
from typing import Dict, List, Optional, Tuple

from config import Config
from medical_texts import NEEDLES, build_corpus_with_needles, estimate_tokens, truncate_to_model_limit
from models import DeepSeekModel, GeminiModel, ModelResponse


SYSTEM_PROMPT = (
    "You are a clinical documentation review assistant. "
    "You will be given a collection of ward round notes. "
    "Answer each question using ONLY information contained in the notes provided. "
    "If the exact answer is present, quote the relevant numbers or phrases directly. "
    "Do not use prior medical knowledge — base your answer solely on the provided notes."
)

QA_PROMPT_TEMPLATE = (
    "The following is a collection of clinical notes from today's ward round.\n"
    "Read all notes carefully, then answer the question at the end.\n\n"
    "{corpus}\n\n"
    "---\n"
    "Question: {question}\n\n"
    "Answer concisely and precisely, quoting the exact value or phrase from the notes."
)


def _build_question_list(needles: List[dict], corpus: str) -> List[Dict]:
    return [
        {
            "id": f"lc-q{n['id']}",
            "prompt": QA_PROMPT_TEMPLATE.format(corpus=corpus, question=n["question"]),
            "needle": n,
        }
        for n in needles
    ]


def prepare_corpora(config: Config) -> Tuple[str, str, str]:
    """
    Build and cache the full corpus, then return per-model truncated versions.
    Returns (full_corpus, deepseek_corpus, gemini_corpus).
    """
    print(f"\nBuilding clinical note corpus (target {config.target_context_tokens:,} tokens)...")
    full_corpus, _ = build_corpus_with_needles(config.target_context_tokens)

    ds_corpus = truncate_to_model_limit(full_corpus, config.deepseek_max_context)
    gm_corpus  = truncate_to_model_limit(full_corpus, config.gemini_max_context)

    print(f"\n  DeepSeek corpus : ~{estimate_tokens(ds_corpus):,} tokens  (limit {config.deepseek_max_context:,})")
    print(f"  Gemini corpus   : ~{estimate_tokens(gm_corpus):,} tokens  (limit {config.gemini_max_context:,})")
    return full_corpus, ds_corpus, gm_corpus


# ---------------------------------------------------------------------------
# Step 1: submit DeepSeek batch (non-blocking)
# ---------------------------------------------------------------------------

def submit_long_context_batch(
    config: Config,
    deepseek: DeepSeekModel,
    ds_corpus: str,
    results_dir: str,
) -> Optional[str]:
    """
    Submit all 5 long-context questions to the Doubleword delayed batch API
    (completion_window=1h). Returns the batch_id so the caller can do other
    work while the job is queued, then call collect_long_context_batch() later.

    The batch_id is also saved to results/deepseek_batch_id.txt for recovery.
    Returns None if batch is not enabled (caller should use sync path instead).
    """
    if not config.use_doubleword_batch:
        return None

    ds_questions = _build_question_list(NEEDLES, ds_corpus)
    print(f"\n  [batch] Submitting {len(ds_questions)} LC questions to Doubleword (1h window)...")

    batch_id = deepseek.submit_batch(
        ds_questions,
        system=SYSTEM_PROMPT,
        max_tokens=512,
    )

    if batch_id:
        id_path = os.path.join(results_dir, "deepseek_batch_id.txt")
        os.makedirs(results_dir, exist_ok=True)
        with open(id_path, "w") as f:
            f.write(batch_id)
        print(f"  [batch] Batch job submitted: {batch_id}  (saved to {id_path})")
        print("  [batch] Continuing with other evaluations — will collect results later.")
    return batch_id


# ---------------------------------------------------------------------------
# Step 2: run Gemini long-context (sync / concurrent)
# ---------------------------------------------------------------------------

def run_gemini_long_context(
    config: Config,
    gemini_flash: GeminiModel,
    gm_corpus: str,
) -> List[ModelResponse]:
    gm_questions = _build_question_list(NEEDLES, gm_corpus)

    if config.use_gemini_batch:
        print(f"\n  Submitting {len(gm_questions)} LC questions to Gemini batch API...")
        return gemini_flash.batch_chat(
            gm_questions,
            system=SYSTEM_PROMPT,
            max_tokens=512,
            poll_interval_s=config.batch_poll_interval_s,
        )
    else:
        print(f"\n  Querying {config.gemini_flash_model} for long-context ({len(gm_questions)} questions)...")
        return gemini_flash.batch_chat(
            gm_questions,
            system=SYSTEM_PROMPT,
            max_tokens=512,
        )


# ---------------------------------------------------------------------------
# Step 3: collect DeepSeek batch results (blocking poll or sync fallback)
# ---------------------------------------------------------------------------

def collect_long_context_batch(
    config: Config,
    deepseek: DeepSeekModel,
    ds_corpus: str,
    batch_id: Optional[str],
    results_dir: str,
) -> List[ModelResponse]:
    """
    If batch_id is given, polls the Doubleword batch API until complete.
    If batch mode is off, runs synchronously instead.
    """
    ds_questions = _build_question_list(NEEDLES, ds_corpus)

    if batch_id:
        print(f"\n  [batch] Collecting DeepSeek batch results (id={batch_id})...")
        return deepseek.collect_batch(
            batch_id=batch_id,
            questions=ds_questions,
            poll_interval_s=config.batch_poll_interval_s,
        )
    else:
        print(f"\n  Querying {config.deepseek_model} for long-context (synchronous)...")
        return [
            deepseek.chat(q["prompt"], system=SYSTEM_PROMPT, max_tokens=512)
            for q in ds_questions
        ]


# ---------------------------------------------------------------------------
# Collate and save results
# ---------------------------------------------------------------------------

def save_long_context_results(
    config: Config,
    ds_corpus: str,
    gm_corpus: str,
    ds_responses: List[ModelResponse],
    gm_responses: List[ModelResponse],
    results_dir: str,
) -> dict:
    print("\n" + "=" * 70)
    print("PART 1: LONG CONTEXT — RESULTS SUMMARY")
    print("=" * 70)

    def _fmt(resp: ModelResponse) -> str:
        if resp.error:
            return f"[ERROR] {resp.error}"
        return (resp.final_text or "[empty]")[:120]

    results = {
        "part": "long_context",
        "deepseek_model": config.deepseek_model,
        "gemini_flash_model": config.gemini_flash_model,
        "deepseek_context_tokens": estimate_tokens(ds_corpus),
        "gemini_context_tokens": estimate_tokens(gm_corpus),
        "deepseek_batch_mode": ds_responses[0].batch_mode if ds_responses else False,
        "gemini_batch_mode": gm_responses[0].batch_mode if gm_responses else False,
        "questions": [],
    }

    for i, needle in enumerate(NEEDLES):
        ds_resp = ds_responses[i]
        gm_resp = gm_responses[i]

        print(
            f"\n  Q{needle['id']} [{needle['position_percent']}%]: {needle['question']}\n"
            f"    Expected  : {needle['expected_answer']}\n"
            f"    DeepSeek  : {_fmt(ds_resp)}\n"
            f"    Gemini    : {_fmt(gm_resp)}"
        )

        results["questions"].append({
            "question_id": needle["id"],
            "position_percent": needle["position_percent"],
            "question": needle["question"],
            "expected_answer": needle["expected_answer"],
            "deepseek": {
                "answer": ds_resp.final_text,
                "latency_s": ds_resp.latency_s,
                "input_tokens": ds_resp.input_tokens,
                "output_tokens": ds_resp.output_tokens,
                "batch_mode": ds_resp.batch_mode,
                "error": ds_resp.error,
            },
            "gemini_flash": {
                "answer": gm_resp.final_text,
                "latency_s": gm_resp.latency_s,
                "input_tokens": gm_resp.input_tokens,
                "output_tokens": gm_resp.output_tokens,
                "batch_mode": gm_resp.batch_mode,
                "error": gm_resp.error,
            },
        })

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "long_context_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nLong context results saved to {out_path}")
    return results
