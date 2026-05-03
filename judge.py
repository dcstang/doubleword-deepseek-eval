"""
Gemini 3 Pro judge for evaluating both parts of the evaluation.

Evaluates:
  a) Long context: accuracy and grounding of needle-in-a-haystack answers
  b) Tool calling: tool selection, parameter correctness, and clinical reasoning quality

Returns structured scores and commentary for each question/scenario.
"""

import json
import os
import time
from typing import Dict

from config import Config
from models import GeminiModel


# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------

LONG_CONTEXT_JUDGE_PROMPT = """You are an expert medical AI evaluator. You will be given:
1. A question about a medical document
2. The expected (correct) answer
3. Two model responses: one from DeepSeek-V4-Flash and one from Gemini-Flash

Your task is to evaluate BOTH responses on:
- ACCURACY (1-5): Does the response contain the exact correct value/fact?
  1=Wrong, 2=Partially correct, 3=Correct but vague, 4=Correct, 5=Exact match with context
- GROUNDEDNESS (1-5): Is the answer clearly based on the document, not hallucinated?
  1=Clearly hallucinated, 2=Likely guessed, 3=Uncertain, 4=Probably grounded, 5=Clearly from text

Respond with valid JSON only, exactly in this format:
{
  "deepseek": {
    "accuracy_score": <int 1-5>,
    "groundedness_score": <int 1-5>,
    "accuracy_justification": "<one sentence>",
    "groundedness_justification": "<one sentence>"
  },
  "gemini_flash": {
    "accuracy_score": <int 1-5>,
    "groundedness_score": <int 1-5>,
    "accuracy_justification": "<one sentence>",
    "groundedness_justification": "<one sentence>"
  },
  "winner": "<deepseek|gemini_flash|tie>",
  "notes": "<optional brief comment>"
}

---
QUESTION: {question}
EXPECTED ANSWER: {expected_answer}
NEEDLE DEPTH IN DOCUMENT: {position_percent}% of the way through

DEEPSEEK RESPONSE:
{deepseek_answer}

GEMINI FLASH RESPONSE:
{gemini_flash_answer}
"""

TOOL_CALLING_JUDGE_PROMPT = """You are an expert clinical AI evaluator. You will assess two AI models on a medical tool-calling task.

Given:
1. A clinical scenario
2. Available tools and which tools were expected to be used
3. Both models' tool calls (name, arguments, results) and final responses
4. A reference clinical answer

Score EACH model on:
- TOOL_SELECTION (0-3): Did it call the right tools?
  0=Called no/wrong tools, 1=Called some but missed key ones, 2=Called all expected tools, 3=Called expected + useful extras
- PARAMETER_ACCURACY (0-3): Were tool arguments clinically correct for the given data?
  0=Incorrect parameters, 1=Partially correct, 2=Mostly correct, 3=All parameters correct
- CLINICAL_REASONING (0-4): Is the final text clinically sound?
  0=Dangerous/wrong, 1=Poor, 2=Adequate, 3=Good, 4=Excellent – safe, accurate, actionable

Respond with valid JSON only, exactly in this format:
{
  "deepseek": {
    "tool_selection_score": <int 0-3>,
    "parameter_accuracy_score": <int 0-3>,
    "clinical_reasoning_score": <int 0-4>,
    "tool_selection_justification": "<one sentence>",
    "parameter_accuracy_justification": "<one sentence>",
    "clinical_reasoning_justification": "<one sentence>"
  },
  "gemini_flash": {
    "tool_selection_score": <int 0-3>,
    "parameter_accuracy_score": <int 0-3>,
    "clinical_reasoning_score": <int 0-4>,
    "tool_selection_justification": "<one sentence>",
    "parameter_accuracy_justification": "<one sentence>",
    "clinical_reasoning_justification": "<one sentence>"
  },
  "winner": "<deepseek|gemini_flash|tie>",
  "safety_flags": "<any dangerous clinical errors, or 'none'>",
  "notes": "<optional brief comment>"
}

---
SCENARIO: {title}

CLINICAL PROMPT:
{prompt}

EXPECTED TOOLS: {expected_tools}
REFERENCE ANSWER: {reference_answer}

DEEPSEEK TOOL CALLS:
{deepseek_tool_calls}

DEEPSEEK FINAL RESPONSE:
{deepseek_final}

GEMINI FLASH TOOL CALLS:
{gemini_flash_tool_calls}

GEMINI FLASH FINAL RESPONSE:
{gemini_flash_final}
"""


def _call_judge(
    judge: GeminiModel,
    prompt: str,
    retries: int = 3,
) -> dict:
    """Call the judge model and parse JSON response with retry on parse failure."""
    for attempt in range(retries):
        resp = judge.chat(
            prompt=prompt,
            system="You are a precise AI evaluator that responds only with valid JSON.",
            max_tokens=1024,
        )
        if resp.error:
            print(f"    Judge error (attempt {attempt + 1}): {resp.error}")
            time.sleep(2 ** attempt)
            continue
        text = resp.final_text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            )
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            print(f"    JSON parse error (attempt {attempt + 1}): {exc}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    return {"error": "Failed to parse judge response after retries", "raw": resp.final_text if resp else ""}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _evaluate_questions(
    judge: GeminiModel,
    questions: list,
) -> tuple:
    """
    Judge a flat list of questions.  Returns (evaluated_list, ds_acc, ds_gnd, gm_acc, gm_gnd, ds_wins, gm_wins, ties).
    """
    evaluated = []
    ds_acc_total = ds_gnd_total = gm_acc_total = gm_gnd_total = 0
    ds_wins = gm_wins = ties = 0

    for q in questions:
        q_id = q["question_id"]
        print(f"\n    Judging Q{q_id} [{q['position_percent']}% depth]...")

        judge_prompt = LONG_CONTEXT_JUDGE_PROMPT.format(
            question=q["question"],
            expected_answer=q["expected_answer"],
            position_percent=q["position_percent"],
            deepseek_answer=(
                q["deepseek"]["answer"]
                if q["deepseek"] and not q["deepseek"].get("error")
                else "[ERROR]"
            ),
            gemini_flash_answer=(
                q["gemini_flash"]["answer"]
                if q["gemini_flash"] and not q["gemini_flash"].get("error")
                else "[ERROR]"
            ),
        )

        judgment = _call_judge(judge, judge_prompt)

        ds_acc = judgment.get("deepseek", {}).get("accuracy_score", 0)
        ds_gnd = judgment.get("deepseek", {}).get("groundedness_score", 0)
        gm_acc = judgment.get("gemini_flash", {}).get("accuracy_score", 0)
        gm_gnd = judgment.get("gemini_flash", {}).get("groundedness_score", 0)

        ds_acc_total += ds_acc
        ds_gnd_total += ds_gnd
        gm_acc_total += gm_acc
        gm_gnd_total += gm_gnd

        winner = judgment.get("winner", "tie")
        if winner == "deepseek":
            ds_wins += 1
        elif winner == "gemini_flash":
            gm_wins += 1
        else:
            ties += 1

        print(f"      DeepSeek    : accuracy={ds_acc}/5  groundedness={ds_gnd}/5")
        print(f"      Gemini Flash: accuracy={gm_acc}/5  groundedness={gm_gnd}/5")
        print(f"      Winner: {winner}")

        evaluated.append({
            "question_id": q_id,
            "position_percent": q["position_percent"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "judgment": judgment,
        })

    return evaluated, ds_acc_total, ds_gnd_total, gm_acc_total, gm_gnd_total, ds_wins, gm_wins, ties


def _make_tier_summary(ds_acc, ds_gnd, gm_acc, gm_gnd, ds_wins, gm_wins, ties, n: int) -> dict:
    n = n or 1
    return {
        "deepseek": {
            "avg_accuracy": round(ds_acc / n, 2),
            "avg_groundedness": round(ds_gnd / n, 2),
            "total_score": round((ds_acc + ds_gnd) / (n * 2) * 100, 1),
            "wins": ds_wins,
        },
        "gemini_flash": {
            "avg_accuracy": round(gm_acc / n, 2),
            "avg_groundedness": round(gm_gnd / n, 2),
            "total_score": round((gm_acc + gm_gnd) / (n * 2) * 100, 1),
            "wins": gm_wins,
        },
        "ties": ties,
        "overall_winner": (
            "deepseek" if ds_wins > gm_wins else
            ("gemini_flash" if gm_wins > ds_wins else "tie")
        ),
    }


def evaluate_long_context(
    config: Config,
    judge: GeminiModel,
    lc_results: dict,
    results_dir: str,
) -> dict:
    """
    Evaluate long-context results.  Handles both the multi-tier format
    (lc_results has 'tiers' key) and the legacy single-tier format
    (lc_results has top-level 'questions' key).
    """
    print("\n" + "=" * 70)
    print("JUDGE: EVALUATING LONG CONTEXT RESULTS")
    print("=" * 70)

    # Multi-tier path
    if "tiers" in lc_results:
        return _evaluate_long_context_tiers(config, judge, lc_results, results_dir)

    # Legacy single-tier path
    evaluated, ds_acc, ds_gnd, gm_acc, gm_gnd, ds_wins, gm_wins, ties = (
        _evaluate_questions(judge, lc_results["questions"])
    )
    n = len(lc_results["questions"])
    evaluation = {
        "part": "long_context_evaluation",
        "judge_model": config.gemini_pro_model,
        "questions": evaluated,
        "summary": _make_tier_summary(ds_acc, ds_gnd, gm_acc, gm_gnd, ds_wins, gm_wins, ties, n),
    }

    print(f"\n  LONG CONTEXT SUMMARY:")
    s = evaluation["summary"]
    print(f"    DeepSeek:     avg accuracy={s['deepseek']['avg_accuracy']}/5, wins={ds_wins}")
    print(f"    Gemini Flash: avg accuracy={s['gemini_flash']['avg_accuracy']}/5, wins={gm_wins}")

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "long_context_evaluation.json")
    with open(out_path, "w") as f:
        json.dump(evaluation, f, indent=2)
    print(f"  Evaluation saved to {out_path}")
    return evaluation


def _evaluate_long_context_tiers(
    config: Config,
    judge: GeminiModel,
    lc_results: dict,
    results_dir: str,
) -> dict:
    """Multi-tier long-context evaluation."""
    evaluation = {
        "part": "long_context_evaluation",
        "judge_model": config.gemini_pro_model,
        "tiers": {},
        "overall_summary": {},
    }

    ds_acc_all = ds_gnd_all = gm_acc_all = gm_gnd_all = 0
    ds_wins_all = gm_wins_all = ties_all = 0
    total_q = 0

    for tier_label, tier_data in lc_results["tiers"].items():
        print(f"\n  --- Tier {tier_label} ---")
        questions = tier_data.get("questions", [])
        evaluated, ds_acc, ds_gnd, gm_acc, gm_gnd, ds_wins, gm_wins, ties = (
            _evaluate_questions(judge, questions)
        )
        n = len(questions)
        tier_summary = _make_tier_summary(
            ds_acc, ds_gnd, gm_acc, gm_gnd, ds_wins, gm_wins, ties, n
        )
        evaluation["tiers"][tier_label] = {
            "questions": evaluated,
            "summary": tier_summary,
        }
        print(
            f"  Tier {tier_label} summary — "
            f"DS avg acc={tier_summary['deepseek']['avg_accuracy']}/5 "
            f"GM avg acc={tier_summary['gemini_flash']['avg_accuracy']}/5 "
            f"winner={tier_summary['overall_winner']}"
        )

        ds_acc_all += ds_acc
        ds_gnd_all += ds_gnd
        gm_acc_all += gm_acc
        gm_gnd_all += gm_gnd
        ds_wins_all += ds_wins
        gm_wins_all += gm_wins
        ties_all += ties
        total_q += n

    evaluation["overall_summary"] = _make_tier_summary(
        ds_acc_all, ds_gnd_all, gm_acc_all, gm_gnd_all,
        ds_wins_all, gm_wins_all, ties_all, total_q,
    )

    print("\n  OVERALL LONG CONTEXT SUMMARY:")
    os_ = evaluation["overall_summary"]
    print(f"    DeepSeek:     avg accuracy={os_['deepseek']['avg_accuracy']}/5, wins={ds_wins_all}")
    print(f"    Gemini Flash: avg accuracy={os_['gemini_flash']['avg_accuracy']}/5, wins={gm_wins_all}")
    print(f"    Overall winner: {os_['overall_winner']}")

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "long_context_evaluation.json")
    with open(out_path, "w") as f:
        json.dump(evaluation, f, indent=2)
    print(f"  Evaluation saved to {out_path}")
    return evaluation


def evaluate_tool_calling(
    config: Config,
    judge: GeminiModel,
    tc_results: dict,
    results_dir: str,
) -> dict:
    print("\n" + "=" * 70)
    print("JUDGE: EVALUATING TOOL CALLING RESULTS")
    print("=" * 70)

    evaluation = {
        "part": "tool_calling_evaluation",
        "judge_model": config.gemini_pro_model,
        "scenarios": [],
        "summary": {},
    }

    ds_tool_sel = ds_param = ds_clinical = 0
    gm_tool_sel = gm_param = gm_clinical = 0
    ds_wins = gm_wins = ties = 0

    for s in tc_results["scenarios"]:
        s_id = s["scenario_id"]
        print(f"\n  Judging Scenario {s_id}: {s['title']}...")

        def fmt_tool_calls(calls):
            if not calls:
                return "No tool calls made"
            lines = []
            for tc in calls:
                lines.append(f"  Tool: {tc['name']}")
                lines.append(f"  Args: {json.dumps(tc['arguments'], indent=4)}")
                lines.append(f"  Result: {json.dumps(tc['result'])[:300]}")
            return "\n".join(lines)

        from scenarios import TOOL_CALLING_SCENARIOS
        scenario_def = next((sc for sc in TOOL_CALLING_SCENARIOS if sc["id"] == s_id), {})

        judge_prompt = TOOL_CALLING_JUDGE_PROMPT.format(
            title=s["title"],
            prompt=scenario_def.get("prompt", ""),
            expected_tools=s["expected_tools"],
            reference_answer=s["reference_answer"],
            deepseek_tool_calls=fmt_tool_calls(s["deepseek"]["tool_calls"] if s["deepseek"] else []),
            deepseek_final=s["deepseek"]["final_text"] if s["deepseek"] and not s["deepseek"].get("error") else "[ERROR]",
            gemini_flash_tool_calls=fmt_tool_calls(s["gemini_flash"]["tool_calls"] if s["gemini_flash"] else []),
            gemini_flash_final=s["gemini_flash"]["final_text"] if s["gemini_flash"] and not s["gemini_flash"].get("error") else "[ERROR]",
        )

        judgment = _call_judge(judge, judge_prompt)

        ds_ts = judgment.get("deepseek", {}).get("tool_selection_score", 0)
        ds_pa = judgment.get("deepseek", {}).get("parameter_accuracy_score", 0)
        ds_cr = judgment.get("deepseek", {}).get("clinical_reasoning_score", 0)
        gm_ts = judgment.get("gemini_flash", {}).get("tool_selection_score", 0)
        gm_pa = judgment.get("gemini_flash", {}).get("parameter_accuracy_score", 0)
        gm_cr = judgment.get("gemini_flash", {}).get("clinical_reasoning_score", 0)

        ds_tool_sel += ds_ts
        ds_param += ds_pa
        ds_clinical += ds_cr
        gm_tool_sel += gm_ts
        gm_param += gm_pa
        gm_clinical += gm_cr

        winner = judgment.get("winner", "tie")
        if winner == "deepseek":
            ds_wins += 1
        elif winner == "gemini_flash":
            gm_wins += 1
        else:
            ties += 1

        print(f"    DeepSeek:     tool_sel={ds_ts}/3  param={ds_pa}/3  clinical={ds_cr}/4")
        print(f"    Gemini Flash: tool_sel={gm_ts}/3  param={gm_pa}/3  clinical={gm_cr}/4")
        print(f"    Winner: {winner}")
        if judgment.get("safety_flags") and judgment["safety_flags"] != "none":
            print(f"    SAFETY FLAGS: {judgment['safety_flags']}")

        evaluation["scenarios"].append({
            "scenario_id": s_id,
            "title": s["title"],
            "judgment": judgment,
        })

    n = len(tc_results["scenarios"]) or 1
    evaluation["summary"] = {
        "deepseek": {
            "avg_tool_selection": round(ds_tool_sel / n, 2),
            "avg_parameter_accuracy": round(ds_param / n, 2),
            "avg_clinical_reasoning": round(ds_clinical / n, 2),
            "total_score": round((ds_tool_sel + ds_param + ds_clinical) / (n * 10) * 100, 1),
            "wins": ds_wins,
        },
        "gemini_flash": {
            "avg_tool_selection": round(gm_tool_sel / n, 2),
            "avg_parameter_accuracy": round(gm_param / n, 2),
            "avg_clinical_reasoning": round(gm_clinical / n, 2),
            "total_score": round((gm_tool_sel + gm_param + gm_clinical) / (n * 10) * 100, 1),
            "wins": gm_wins,
        },
        "ties": ties,
        "overall_winner": "deepseek" if ds_wins > gm_wins else ("gemini_flash" if gm_wins > ds_wins else "tie"),
    }

    print(f"\n  TOOL CALLING SUMMARY:")
    print(f"    DeepSeek:     total score={evaluation['summary']['deepseek']['total_score']}%,  wins={ds_wins}")
    print(f"    Gemini Flash: total score={evaluation['summary']['gemini_flash']['total_score']}%, wins={gm_wins}")

    # Save
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "tool_calling_evaluation.json")
    with open(out_path, "w") as f:
        json.dump(evaluation, f, indent=2)
    print(f"  Evaluation saved to {out_path}")

    return evaluation
