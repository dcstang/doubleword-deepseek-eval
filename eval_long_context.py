"""
Part 1: Long-context needle-in-a-haystack evaluation.

Builds a medical text corpus from HuggingFace (up to 800k tokens), plants 5 specific
clinical facts at different depth positions, then asks each model 5 QA questions.

When batch mode is enabled:
  - DeepSeek: submits all 5 questions in one Doubleword delayed batch (completion_window=1h)
  - Gemini:   submits via google-genai batch API or concurrent asyncio fallback
"""

import json
import os
from typing import List

from config import Config
from medical_texts import NEEDLES, build_corpus_with_needles, estimate_tokens, truncate_to_model_limit
from models import DeepSeekModel, GeminiModel, ModelResponse


SYSTEM_PROMPT = (
    "You are a medical information retrieval assistant. "
    "A long medical document will be provided. "
    "Answer each question using ONLY information contained in the document. "
    "If the exact answer is present, quote the relevant numbers or facts directly. "
    "Do not guess or use prior knowledge – base your answer solely on the provided text."
)

QA_PROMPT_TEMPLATE = (
    "Below is a long medical document. Read it carefully, then answer the question.\n\n"
    "=== MEDICAL DOCUMENT ===\n"
    "{corpus}\n"
    "=== END OF DOCUMENT ===\n\n"
    "Question: {question}\n\n"
    "Answer concisely and precisely. Quote the exact value or phrase if it appears in the document."
)


def run_long_context_eval(
    config: Config,
    deepseek: DeepSeekModel,
    gemini_flash: GeminiModel,
    results_dir: str,
) -> dict:
    print("\n" + "=" * 70)
    print("PART 1: LONG CONTEXT NEEDLE-IN-A-HAYSTACK EVALUATION")
    print("=" * 70)

    # Build shared corpus then truncate per model
    print(f"\nBuilding medical corpus (target {config.target_context_tokens:,} tokens)...")
    full_corpus, needles = build_corpus_with_needles(config.target_context_tokens)

    deepseek_corpus = truncate_to_model_limit(full_corpus, config.deepseek_max_context)
    gemini_corpus   = truncate_to_model_limit(full_corpus, config.gemini_max_context)

    ds_ctx_tokens = estimate_tokens(deepseek_corpus)
    gm_ctx_tokens = estimate_tokens(gemini_corpus)
    print(f"\nDeepSeek corpus : ~{ds_ctx_tokens:,} tokens  (limit {config.deepseek_max_context:,})")
    print(f"Gemini corpus   : ~{gm_ctx_tokens:,} tokens  (limit {config.gemini_max_context:,})")
    print(f"\nBatch mode — DeepSeek: {config.use_doubleword_batch}  |  Gemini: {config.use_gemini_batch}")

    results = {
        "part": "long_context",
        "deepseek_model": config.deepseek_model,
        "gemini_flash_model": config.gemini_flash_model,
        "deepseek_context_tokens": ds_ctx_tokens,
        "gemini_context_tokens": gm_ctx_tokens,
        "deepseek_batch_mode": config.use_doubleword_batch,
        "gemini_batch_mode": config.use_gemini_batch,
        "questions": [],
    }

    # Build per-model question lists for batch submission
    ds_questions = [
        {
            "id": f"lc-q{n['id']}",
            "prompt": QA_PROMPT_TEMPLATE.format(corpus=deepseek_corpus, question=n["question"]),
            "needle": n,
        }
        for n in needles
    ]
    gm_questions = [
        {
            "id": f"lc-q{n['id']}",
            "prompt": QA_PROMPT_TEMPLATE.format(corpus=gemini_corpus, question=n["question"]),
            "needle": n,
        }
        for n in needles
    ]

    # ---------- DeepSeek ----------
    if config.use_doubleword_batch:
        print(f"\n  Submitting {len(ds_questions)} questions to Doubleword batch (1h window)...")
        ds_responses = deepseek.batch_chat(
            ds_questions, system=SYSTEM_PROMPT, max_tokens=512,
            poll_interval_s=config.batch_poll_interval_s,
        )
    else:
        print(f"\n  Querying {config.deepseek_model} synchronously...")
        ds_responses = [
            deepseek.chat(q["prompt"], system=SYSTEM_PROMPT, max_tokens=512)
            for q in ds_questions
        ]

    # ---------- Gemini Flash ----------
    if config.use_gemini_batch:
        print(f"\n  Submitting {len(gm_questions)} questions to Gemini batch API...")
        gm_responses = gemini_flash.batch_chat(
            gm_questions, system=SYSTEM_PROMPT, max_tokens=512,
            poll_interval_s=config.batch_poll_interval_s,
        )
    else:
        print(f"\n  Querying {config.gemini_flash_model} synchronously...")
        gm_responses = [
            gemini_flash.chat(q["prompt"], system=SYSTEM_PROMPT, max_tokens=512)
            for q in gm_questions
        ]

    # Collate results
    for i, needle in enumerate(needles):
        ds_resp = ds_responses[i]
        gm_resp = gm_responses[i]
        print(
            f"\n  Q{needle['id']} [{needle['position_percent']}%]: {needle['question']}\n"
            f"    Expected  : {needle['expected_answer']}\n"
            f"    DeepSeek  : {(ds_resp.final_text or '[ERROR]')[:120]}\n"
            f"    Gemini    : {(gm_resp.final_text or '[ERROR]')[:120]}"
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
