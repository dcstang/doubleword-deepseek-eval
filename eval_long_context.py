"""
Part 1: Long-context needle-in-a-haystack evaluation — multi-tier.

Corpus: numbered clinical notes from AGBonnet/augmented-clinical-notes with 5 needle
notes planted at specific depth positions (5/25/50/75/95%).

Four context tiers: 200k / 400k / 600k / 800k tokens.  Each tier uses a
different random seed so the background notes differ between tiers, but both
models see exactly the same notes at each tier.

Batch flow (Doubleword delayed, 1h window):
  1. submit_all_tiers_batch()  — upload all 4×5=20 questions, returns batch_id
  2. <caller runs Gemini + tool-calling while the job queues>
  3. collect_all_tiers_batch() — polls until done, returns per-tier responses

Gemini runs synchronously per tier (or concurrently via asyncio fallback).
"""

import json
import os
from typing import Dict, List, Optional, Tuple

from config import Config
from medical_texts import (
    CONTEXT_TIERS,
    TIER_SEEDS,
    NEEDLES,
    build_corpus_with_needles,
    estimate_tokens,
    truncate_to_model_limit,
)
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


def _tier_label(tier: int) -> str:
    return f"{tier // 1000}k"


def _build_tier_question_list(
    needles: List[dict],
    corpus: str,
    tier: int,
    q_prefix: str = "lc",
) -> List[Dict]:
    label = _tier_label(tier)
    return [
        {
            "id": f"{q_prefix}-t{label}-q{n['id']}",
            "prompt": QA_PROMPT_TEMPLATE.format(corpus=corpus, question=n["question"]),
            "needle": n,
            "tier": tier,
        }
        for n in needles
    ]


# ---------------------------------------------------------------------------
# Step 1: build / load corpora for all tiers
# ---------------------------------------------------------------------------

def prepare_all_tier_corpora(config: Config) -> Dict[int, Tuple[str, str]]:
    """
    Build or load cached corpora for all context tiers.
    Returns {tier_tokens: (ds_corpus, gm_corpus)}.
    Each tier uses its own cache file and random seed so content differs.
    """
    tier_corpora: Dict[int, Tuple[str, str]] = {}

    for tier in CONTEXT_TIERS:
        seed = TIER_SEEDS[tier]
        label = _tier_label(tier)
        print(f"\nBuilding {label} corpus (seed={seed}, target={tier:,} tokens)...")

        full_corpus, _ = build_corpus_with_needles(tier, seed=seed)

        ds_corpus = truncate_to_model_limit(full_corpus, config.deepseek_max_context)
        gm_corpus = truncate_to_model_limit(full_corpus, config.gemini_max_context)

        print(
            f"  DeepSeek corpus : ~{estimate_tokens(ds_corpus):,} tokens"
            f"  (limit {config.deepseek_max_context:,})"
        )
        print(
            f"  Gemini corpus   : ~{estimate_tokens(gm_corpus):,} tokens"
            f"  (limit {config.gemini_max_context:,})"
        )

        tier_corpora[tier] = (ds_corpus, gm_corpus)

    return tier_corpora


# ---------------------------------------------------------------------------
# Step 2a: submit ALL tier questions in one DeepSeek batch (non-blocking)
# ---------------------------------------------------------------------------

def submit_all_tiers_batch(
    config: Config,
    deepseek: DeepSeekModel,
    tier_corpora: Dict[int, Tuple[str, str]],
    results_dir: str,
    q_prefix: str = "lc",
    batch_id_filename: str = "deepseek_batch_id.txt",
) -> Optional[str]:
    """
    Submit all 4×5=20 long-context questions in a single Doubleword batch
    (completion_window=1h).  Returns batch_id immediately.

    q_prefix: prefix for question IDs — use "lc" for Flash, "lc-pro" for Pro.
    batch_id_filename: file to persist the batch_id for recovery.
    Returns None if batch mode is disabled (synchronous fallback used later).
    """
    if not config.use_doubleword_batch:
        return None

    # Resume from a previous submission if the batch_id file already exists
    id_path = os.path.join(results_dir, batch_id_filename)
    if os.path.exists(id_path):
        with open(id_path) as f:
            existing_id = f.read().strip()
        if existing_id:
            print(f"\n  [batch] Reusing existing batch job {existing_id} (from {id_path})")
            return existing_id

    all_questions: List[Dict] = []
    for tier in CONTEXT_TIERS:
        ds_corpus, _ = tier_corpora[tier]
        all_questions.extend(_build_tier_question_list(NEEDLES, ds_corpus, tier, q_prefix))

    n_tiers = len(CONTEXT_TIERS)
    print(
        f"\n  [batch] Submitting {len(all_questions)} LC questions "
        f"({n_tiers} tiers × 5, prefix={q_prefix}) to Doubleword (1h window)..."
    )

    batch_id = deepseek.submit_batch(
        all_questions,
        system=SYSTEM_PROMPT,
        max_tokens=512,
    )

    if batch_id:
        os.makedirs(results_dir, exist_ok=True)
        with open(id_path, "w") as f:
            f.write(batch_id)
        print(f"  [batch] Batch job submitted: {batch_id}  (saved to {id_path})")
        print("  [batch] Continuing with Gemini + tool-calling — will collect later.")

    return batch_id


# ---------------------------------------------------------------------------
# Step 2b: run Gemini for all tiers (sync per tier, async within tier)
# ---------------------------------------------------------------------------

def run_all_gemini_tiers(
    config: Config,
    gemini_flash: GeminiModel,
    tier_corpora: Dict[int, Tuple[str, str]],
    use_batch: bool = None,   # None → read from config.use_gemini_batch
) -> Dict[int, List[ModelResponse]]:
    """
    Run Gemini on all context tiers in sequence.
    Returns {tier_tokens: [ModelResponse × 5]}.
    """
    tier_responses: Dict[int, List[ModelResponse]] = {}

    for tier in CONTEXT_TIERS:
        _, gm_corpus = tier_corpora[tier]
        questions = _build_tier_question_list(NEEDLES, gm_corpus, tier)
        label = _tier_label(tier)

        print(
            f"\n  Querying {config.gemini_flash_model} — tier {label} "
            f"({len(questions)} questions)..."
        )

        do_batch = config.use_gemini_batch if use_batch is None else use_batch
        if do_batch:
            responses = gemini_flash.batch_chat(
                questions,
                system=SYSTEM_PROMPT,
                max_tokens=512,
                poll_interval_s=config.batch_poll_interval_s,
            )
        else:
            responses = gemini_flash.batch_chat(
                questions,
                system=SYSTEM_PROMPT,
                max_tokens=512,
            )

        tier_responses[tier] = responses

    return tier_responses


# ---------------------------------------------------------------------------
# Step 3: collect DeepSeek batch results (or run synchronously)
# ---------------------------------------------------------------------------

def collect_all_tiers_batch(
    config: Config,
    deepseek: DeepSeekModel,
    tier_corpora: Dict[int, Tuple[str, str]],
    batch_id: Optional[str],
    results_dir: str,
    q_prefix: str = "lc",
) -> Dict[int, List[ModelResponse]]:
    """
    If batch_id given, polls Doubleword until all 20 results are ready.
    Otherwise runs all questions synchronously.
    Returns {tier_tokens: [ModelResponse × 5]}.
    """
    all_questions: List[Dict] = []
    for tier in CONTEXT_TIERS:
        ds_corpus, _ = tier_corpora[tier]
        all_questions.extend(_build_tier_question_list(NEEDLES, ds_corpus, tier, q_prefix))

    if batch_id:
        print(
            f"\n  [batch] Collecting {len(all_questions)} DeepSeek LC responses "
            f"(id={batch_id})..."
        )
        all_responses = deepseek.collect_batch(
            batch_id=batch_id,
            questions=all_questions,
            poll_interval_s=config.batch_poll_interval_s,
        )
    else:
        print(
            f"\n  Querying {config.deepseek_model} synchronously "
            f"({len(all_questions)} LC questions)..."
        )
        all_responses = [
            deepseek.chat(q["prompt"], system=SYSTEM_PROMPT, max_tokens=512)
            for q in all_questions
        ]

    # Split flat list back into per-tier buckets (order preserved from build loop)
    per_tier = len(NEEDLES)
    tier_responses: Dict[int, List[ModelResponse]] = {}
    for i, tier in enumerate(CONTEXT_TIERS):
        tier_responses[tier] = all_responses[i * per_tier : (i + 1) * per_tier]

    return tier_responses


# ---------------------------------------------------------------------------
# Collate and save results
# ---------------------------------------------------------------------------

def save_all_tier_results(
    config: Config,
    tier_corpora: Dict[int, Tuple[str, str]],
    ds_tier_responses: Dict[int, List[ModelResponse]],
    gm_tier_responses: Dict[int, List[ModelResponse]],
    results_dir: str,
    save_suffix: str = "",
    deepseek_label: Optional[str] = None,
    gemini_label: Optional[str] = None,
) -> dict:
    print("\n" + "=" * 70)
    print("PART 1: LONG CONTEXT — RESULTS SUMMARY")
    print("=" * 70)

    def _fmt(resp: Optional[ModelResponse]) -> str:
        if resp is None:
            return "[missing]"
        if resp.error:
            return f"[ERROR] {resp.error}"
        return (resp.final_text or "[empty]")[:120]

    results = {
        "part": "long_context",
        "deepseek_model": deepseek_label or config.deepseek_model,
        "gemini_flash_model": gemini_label or config.gemini_flash_model,
        "tiers": {},
    }

    for tier in CONTEXT_TIERS:
        label = _tier_label(tier)
        ds_corpus, gm_corpus = tier_corpora[tier]
        ds_responses = ds_tier_responses.get(tier, [])
        gm_responses = gm_tier_responses.get(tier, [])

        tier_data: dict = {
            "tier_tokens": tier,
            "deepseek_context_tokens": estimate_tokens(ds_corpus),
            "gemini_context_tokens": estimate_tokens(gm_corpus),
            "deepseek_batch_mode": ds_responses[0].batch_mode if ds_responses else False,
            "gemini_batch_mode": gm_responses[0].batch_mode if gm_responses else False,
            "questions": [],
        }

        print(f"\n  --- Tier {label} ---")

        for i, needle in enumerate(NEEDLES):
            ds_resp = ds_responses[i] if i < len(ds_responses) else None
            gm_resp = gm_responses[i] if i < len(gm_responses) else None

            print(
                f"\n  Q{needle['id']} [{needle['position_percent']}%]: {needle['question']}\n"
                f"    Expected : {needle['expected_answer']}\n"
                f"    DeepSeek : {_fmt(ds_resp)}\n"
                f"    Gemini   : {_fmt(gm_resp)}"
            )

            def _resp_dict(resp: Optional[ModelResponse]) -> Optional[dict]:
                if resp is None:
                    return None
                return {
                    "answer": resp.final_text,
                    "latency_s": resp.latency_s,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "batch_mode": resp.batch_mode,
                    "error": resp.error,
                }

            tier_data["questions"].append({
                "question_id": needle["id"],
                "position_percent": needle["position_percent"],
                "question": needle["question"],
                "expected_answer": needle["expected_answer"],
                "deepseek": _resp_dict(ds_resp),
                "gemini_flash": _resp_dict(gm_resp),
            })

        results["tiers"][label] = tier_data

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"long_context_results{save_suffix}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nLong context results saved to {out_path}")
    return results
