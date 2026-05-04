#!/usr/bin/env python3
"""
DeepSeek-V4-Flash vs Gemini 3 Flash — Medical Evaluation

Two-part benchmark:
  Part 1 — Long context (needle-in-a-haystack):
    Four tiers (200k / 400k / 600k / 800k tokens) of clinical notes from
    HuggingFace with 5 planted facts at depths 5%, 25%, 50%, 75%, 95%.
    Each tier uses a different random seed (different notes) but both models
    see the same notes at every tier.

  Part 2 — Tool calling:
    5 clinical scenarios (sepsis, ICU scoring, DVT, TBI, AECOPD) with medical
    scoring tools (SOFA, APACHE II, GCS, Wells DVT, qSOFA, BMI) plus general
    utilities. Each model must select and call the right tools, then give a
    clinical recommendation.

  Evaluation:
    Gemini 3 Pro judges both parts and produces a final report with scores,
    cost comparison, and head-to-head verdict.

Usage:
    cp .env.example .env   # fill in your API keys
    pip install -r requirements.txt
    python main.py [--skip-long-context] [--skip-tool-calling] [--skip-judge]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

from config import load_config
from eval_long_context import (
    prepare_all_tier_corpora,
    submit_all_tiers_batch,
    run_all_gemini_tiers,
    collect_all_tiers_batch,
    save_all_tier_results,
)
from eval_tool_calling import run_tool_calling_eval
from judge import evaluate_long_context, evaluate_tool_calling
from models import DeepSeekModel, GeminiModel


# ---------------------------------------------------------------------------
# Cost summary
# ---------------------------------------------------------------------------

def compute_cost_summary(
    config,
    lc_results: dict,
    tc_results: dict,
    lc_eval: dict,
    tc_eval: dict,
    lc_results_pro: dict = None,
    tc_results_pro: dict = None,
) -> dict:
    """Aggregate token usage and compute estimated costs with/without batch discount."""

    def tally(results_part: dict, model_key: str):
        inp = out = 0
        batch = False
        if results_part is None:
            return inp, out, batch
        # Multi-tier long-context format: tiers -> {label -> {questions: [...]}}
        if "tiers" in results_part:
            for tier_data in results_part["tiers"].values():
                for item in tier_data.get("questions", []):
                    r = item.get(model_key) or {}
                    inp += r.get("input_tokens", 0)
                    out += r.get("output_tokens", 0)
                    if r.get("batch_mode"):
                        batch = True
        else:
            for item in results_part.get("questions", results_part.get("scenarios", [])):
                r = item.get(model_key, {}) or {}
                inp += r.get("input_tokens", 0)
                out += r.get("output_tokens", 0)
                if r.get("batch_mode"):
                    batch = True
        return inp, out, batch

    # Eval tokens (judge, always sync)
    def tally_judge(eval_part: dict):
        inp = out = 0
        for item in eval_part.get("questions", eval_part.get("scenarios", [])):
            j = item.get("judgment", {})
            # judge token counts aren't tracked per-question; estimate later
        return inp, out

    ds_lc_in,  ds_lc_out,  ds_lc_batch  = tally(lc_results, "deepseek")
    ds_tc_in,  ds_tc_out,  _            = tally(tc_results, "deepseek")
    gm_lc_in,  gm_lc_out,  gm_lc_batch  = tally(lc_results, "gemini_flash")
    gm_tc_in,  gm_tc_out,  _            = tally(tc_results, "gemini_flash")

    dsp_lc_in, dsp_lc_out, dsp_lc_batch = tally(lc_results_pro, "deepseek")
    dsp_tc_in, dsp_tc_out, _            = tally(tc_results_pro, "deepseek")
    gmp_lc_in, gmp_lc_out, gmp_lc_batch = tally(lc_results_pro, "gemini_flash")
    gmp_tc_in, gmp_tc_out, _            = tally(tc_results_pro, "gemini_flash")

    p_ds  = config.deepseek_pricing
    p_gm  = config.gemini_flash_pricing
    p_dsp = config.deepseek_pro_pricing
    p_gmp = config.gemini_eval_pro_pricing
    p_judge = config.gemini_pro_pricing

    def _model_cost_block(p, lc_in, lc_out, lc_batch, tc_in, tc_out, model_name):
        total_in  = lc_in  + tc_in
        total_out = lc_out + tc_out
        standard  = p.cost(total_in, total_out, batch=False)
        lc_actual = p.cost(lc_in, lc_out, batch=lc_batch)
        tc_actual = p.cost(tc_in, tc_out, batch=False)
        actual    = lc_actual + tc_actual
        return {
            "model": model_name,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "long_context_batch_mode": lc_batch,
            "cost_standard_usd": round(standard, 4),
            "cost_actual_usd":   round(actual, 4),
            "savings_usd":       round(standard - actual, 4),
            "pricing": {
                "input_per_million":   p.input_per_million,
                "output_per_million":  p.output_per_million,
                "batch_discount_pct":  int(p.batch_discount * 100),
            },
        }

    # Judge: ~2k in / 0.5k out per judgment × 10 judgments (×2 if pro run)
    n_judge_sets = 2 if lc_results_pro else 1
    judge_in_est  = n_judge_sets * 10 * 2000
    judge_out_est = n_judge_sets * 10 * 500
    judge_cost    = p_judge.cost(judge_in_est, judge_out_est)

    ds_block  = _model_cost_block(p_ds,  ds_lc_in,  ds_lc_out,  ds_lc_batch,  ds_tc_in,  ds_tc_out,  config.deepseek_model)
    gm_block  = _model_cost_block(p_gm,  gm_lc_in,  gm_lc_out,  gm_lc_batch,  gm_tc_in,  gm_tc_out,  config.gemini_flash_model)
    dsp_block = _model_cost_block(p_dsp, dsp_lc_in, dsp_lc_out, dsp_lc_batch, dsp_tc_in, dsp_tc_out, config.deepseek_pro_model or "")
    gmp_block = _model_cost_block(p_gmp, gmp_lc_in, gmp_lc_out, gmp_lc_batch, gmp_tc_in, gmp_tc_out, config.gemini_eval_pro_model or "")

    total_standard = sum(b["cost_standard_usd"] for b in [ds_block, gm_block, dsp_block, gmp_block]) + judge_cost
    total_actual   = sum(b["cost_actual_usd"]   for b in [ds_block, gm_block, dsp_block, gmp_block]) + judge_cost
    savings        = total_standard - total_actual

    out = {
        "deepseek_flash": ds_block,
        "gemini_flash":   gm_block,
        "judge_gemini_pro": {
            "model": config.gemini_pro_model,
            "estimated_input_tokens": judge_in_est,
            "estimated_output_tokens": judge_out_est,
            "estimated_cost_usd": round(judge_cost, 4),
        },
        "totals": {
            "cost_without_batch_usd": round(total_standard, 4),
            "cost_with_batch_usd":    round(total_actual, 4),
            "batch_savings_usd":      round(savings, 4),
            "batch_savings_pct":      round(savings / total_standard * 100, 1) if total_standard > 0 else 0,
        },
        "note": "Pricing estimates — verify at doubleword.ai and ai.google.dev/pricing",
    }
    if lc_results_pro:
        out["deepseek_pro"]      = dsp_block
        out["gemini_eval_pro"]   = gmp_block
    return out


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

def write_report(
    config,
    lc_results: dict,
    tc_results: dict,
    lc_eval: dict,
    tc_eval: dict,
    cost_summary: dict,
    results_dir: str,
    lc_results_pro: dict = None,
    tc_results_pro: dict = None,
    lc_eval_pro: dict = None,
    tc_eval_pro: dict = None,
) -> str:
    # Multi-tier uses overall_summary; legacy single-tier uses summary
    lc_overall = lc_eval.get("overall_summary", lc_eval.get("summary", {}))
    tc_sum = tc_eval.get("summary", {})
    ds_lc = lc_overall.get("deepseek", {})
    gm_lc = lc_overall.get("gemini_flash", {})
    ds_tc = tc_sum.get("deepseek", {})
    gm_tc = tc_sum.get("gemini_flash", {})

    has_pro = bool(lc_results_pro or tc_results_pro)
    lc_overall_pro = (lc_eval_pro or {}).get("overall_summary", (lc_eval_pro or {}).get("summary", {}))
    tc_sum_pro     = (tc_eval_pro or {}).get("summary", {})
    dsp_lc = lc_overall_pro.get("deepseek", {})
    gmp_lc = lc_overall_pro.get("gemini_flash", {})
    dsp_tc = tc_sum_pro.get("deepseek", {})
    gmp_tc = tc_sum_pro.get("gemini_flash", {})

    totals  = cost_summary["totals"]

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    multi_tier = "tiers" in lc_eval

    lines = [
        f"# DeepSeek-V4-Flash vs Gemini 3 Flash — Medical Evaluation Report",
        f"*Generated: {now}*",
        "",
        "## Models",
        f"| Role | Model |",
        f"|------|-------|",
        f"| Evaluated | `{config.deepseek_model}` (via Doubleword) |",
        f"| Evaluated | `{config.gemini_flash_model}` (Google Gemini API) |",
        f"| Judge     | `{config.gemini_pro_model}` |",
        "",
        "---",
        "",
        "## Part 1 — Long Context (Needle-in-a-Haystack)",
        "5 clinical facts planted at depths: 5%, 25%, 50%, 75%, 95%",
        "Four context tiers: 200k / 400k / 600k / 800k tokens  "
        "(each tier uses a different random seed for varied background notes)",
        "",
    ]

    if multi_tier:
        for tier_label, tier_eval in lc_eval.get("tiers", {}).items():
            tier_result = lc_results.get("tiers", {}).get(tier_label, {})
            ds_ctx = tier_result.get("deepseek_context_tokens", 0)
            gm_ctx = tier_result.get("gemini_context_tokens", 0)
            ts = tier_eval.get("summary", {})
            ds_ts = ts.get("deepseek", {})
            gm_ts = ts.get("gemini_flash", {})

            lines += [
                f"### Tier {tier_label}",
                f"Context: ~{ds_ctx:,} tokens (DeepSeek) / ~{gm_ctx:,} tokens (Gemini)",
                "",
                "| Q | Depth | DS Acc | DS Gnd | GM Acc | GM Gnd | Winner |",
                "|---|-------|:---:|:---:|:---:|:---:|:---:|",
            ]
            for q in tier_eval.get("questions", []):
                j = q.get("judgment", {})
                ds_j = j.get("deepseek", {})
                gm_j = j.get("gemini_flash", {})
                winner = j.get("winner", "?")
                lines.append(
                    f"| Q{q['question_id']} | {q['position_percent']}% "
                    f"| {ds_j.get('accuracy_score','?')}/5 | {ds_j.get('groundedness_score','?')}/5 "
                    f"| {gm_j.get('accuracy_score','?')}/5 | {gm_j.get('groundedness_score','?')}/5 "
                    f"| **{winner}** |"
                )
            lines += [
                "",
                f"Tier {tier_label} — DS score {ds_ts.get('total_score','?')}% / "
                f"GM score {gm_ts.get('total_score','?')}%  "
                f"winner: **{ts.get('overall_winner','?')}**",
                "",
            ]
    else:
        lines += [
            "### Per-Question Scores (Accuracy / Groundedness out of 5)",
            "| Q | Depth | DeepSeek Acc | DeepSeek Gnd | Gemini Acc | Gemini Gnd | Winner |",
            "|---|-------|:---:|:---:|:---:|:---:|:---:|",
        ]
        for q in lc_eval.get("questions", []):
            j = q.get("judgment", {})
            ds_j = j.get("deepseek", {})
            gm_j = j.get("gemini_flash", {})
            winner = j.get("winner", "?")
            lines.append(
                f"| {q['question_id']} | {q['position_percent']}% "
                f"| {ds_j.get('accuracy_score','?')}/5 | {ds_j.get('groundedness_score','?')}/5 "
                f"| {gm_j.get('accuracy_score','?')}/5 | {gm_j.get('groundedness_score','?')}/5 "
                f"| **{winner}** |"
            )
        lines.append("")

    lines += [
        "### Long Context Overall Summary",
        f"| Model | Avg Accuracy | Avg Groundedness | Score | Wins |",
        f"|-------|:---:|:---:|:---:|:---:|",
        f"| DeepSeek-V4-Flash | {ds_lc.get('avg_accuracy','?')}/5 | {ds_lc.get('avg_groundedness','?')}/5 | {ds_lc.get('total_score','?')}% | {ds_lc.get('wins','?')} |",
        f"| Gemini 3 Flash    | {gm_lc.get('avg_accuracy','?')}/5 | {gm_lc.get('avg_groundedness','?')}/5 | {gm_lc.get('total_score','?')}% | {gm_lc.get('wins','?')} |",
        f"",
        f"**Long context winner: {lc_overall.get('overall_winner', '?').upper()}**",
        "",
        "---",
        "",
        "## Part 2 — Tool Calling",
        "5 medical scenarios: Septic Shock, Post-Cardiac Surgery ICU, DVT, TBI, AECOPD",
        "Tools available: qSOFA, SOFA, APACHE II, GCS, Wells DVT, BMI, convert_units, drug_interactions, guidelines",
        "",
        "### Per-Scenario Scores (Tool Selection / Param Accuracy / Clinical Reasoning)",
        "| # | Scenario | DS Tool | DS Param | DS Clin | GM Tool | GM Param | GM Clin | Winner |",
        "|---|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for s in tc_eval.get("scenarios", []):
        j = s.get("judgment", {})
        ds_j = j.get("deepseek", {})
        gm_j = j.get("gemini_flash", {})
        winner = j.get("winner", "?")
        safety = j.get("safety_flags", "none")
        flag = " ⚠️" if safety and safety != "none" else ""
        lines.append(
            f"| {s['scenario_id']} | {s['title']}{flag} "
            f"| {ds_j.get('tool_selection_score','?')}/3 | {ds_j.get('parameter_accuracy_score','?')}/3 | {ds_j.get('clinical_reasoning_score','?')}/4 "
            f"| {gm_j.get('tool_selection_score','?')}/3 | {gm_j.get('parameter_accuracy_score','?')}/3 | {gm_j.get('clinical_reasoning_score','?')}/4 "
            f"| **{winner}** |"
        )

    ds_flash_label  = lc_results.get("deepseek_model", config.deepseek_model)
    gm_flash_label  = lc_results.get("gemini_flash_model", config.gemini_flash_model)
    ds_pro_label    = (lc_results_pro or {}).get("deepseek_model", config.deepseek_pro_model or "DS Pro")
    gm_pro_label    = (lc_results_pro or {}).get("gemini_flash_model", config.gemini_eval_pro_model or "GM Pro")

    tc_hdr = "| # | Scenario | DS-F Tool | DS-F Param | DS-F Clin | GM-F Tool | GM-F Param | GM-F Clin |"
    tc_sep = "|---|----------|:---:|:---:|:---:|:---:|:---:|:---:|"
    if has_pro:
        tc_hdr += " DSP Tool | DSP Param | DSP Clin | GMP Tool | GMP Param | GMP Clin | "
        tc_sep += ":---:|:---:|:---:|:---:|:---:|:---:|"
    tc_hdr += " Winner |"
    tc_sep += ":---:|"

    tc_pro_scenarios = {s["scenario_id"]: s for s in (tc_eval_pro or {}).get("scenarios", [])}

    lines += [
        "",
        "### Tool Calling Summary",
        f"| Model | Avg Tool Sel | Avg Params | Avg Clinical | Score | Wins |",
        f"|-------|:---:|:---:|:---:|:---:|:---:|",
        f"| {ds_flash_label} | {ds_tc.get('avg_tool_selection','?')}/3 | {ds_tc.get('avg_parameter_accuracy','?')}/3 | {ds_tc.get('avg_clinical_reasoning','?')}/4 | {ds_tc.get('total_score','?')}% | {ds_tc.get('wins','?')} |",
        f"| {gm_flash_label} | {gm_tc.get('avg_tool_selection','?')}/3 | {gm_tc.get('avg_parameter_accuracy','?')}/3 | {gm_tc.get('avg_clinical_reasoning','?')}/4 | {gm_tc.get('total_score','?')}% | {gm_tc.get('wins','?')} |",
    ]
    if has_pro:
        lines += [
            f"| {ds_pro_label} | {dsp_tc.get('avg_tool_selection','?')}/3 | {dsp_tc.get('avg_parameter_accuracy','?')}/3 | {dsp_tc.get('avg_clinical_reasoning','?')}/4 | {dsp_tc.get('total_score','?')}% | {dsp_tc.get('wins','?')} |",
            f"| {gm_pro_label} | {gmp_tc.get('avg_tool_selection','?')}/3 | {gmp_tc.get('avg_parameter_accuracy','?')}/3 | {gmp_tc.get('avg_clinical_reasoning','?')}/4 | {gmp_tc.get('total_score','?')}% | {gmp_tc.get('wins','?')} |",
        ]
    lines += [
        "",
        f"**Tool calling winner (Flash): {tc_sum.get('overall_winner', '?').upper()}**",
    ]
    if has_pro:
        lines.append(f"**Tool calling winner (Pro): {tc_sum_pro.get('overall_winner', '?').upper()}**")
    lines += [
        "",
        "---",
        "",
        "## Cost Comparison",
        "",
    ]

    # Cost rows for all active models
    ds_block  = cost_summary.get("deepseek_flash", {})
    gm_block  = cost_summary.get("gemini_flash", {})
    dsp_block = cost_summary.get("deepseek_pro", {})
    gmp_block = cost_summary.get("gemini_eval_pro", {})

    cost_rows = [
        f"| {ds_flash_label} | {ds_block.get('total_input_tokens',0):,} | {ds_block.get('total_output_tokens',0):,} | ${ds_block.get('cost_standard_usd',0):.4f} | ${ds_block.get('cost_actual_usd',0):.4f} | ${ds_block.get('savings_usd',0):.4f} | {ds_block.get('long_context_batch_mode',False)} |",
        f"| {gm_flash_label} | {gm_block.get('total_input_tokens',0):,} | {gm_block.get('total_output_tokens',0):,} | ${gm_block.get('cost_standard_usd',0):.4f} | ${gm_block.get('cost_actual_usd',0):.4f} | ${gm_block.get('savings_usd',0):.4f} | {gm_block.get('long_context_batch_mode',False)} |",
    ]
    if has_pro and dsp_block:
        cost_rows.append(f"| {ds_pro_label} | {dsp_block.get('total_input_tokens',0):,} | {dsp_block.get('total_output_tokens',0):,} | ${dsp_block.get('cost_standard_usd',0):.4f} | ${dsp_block.get('cost_actual_usd',0):.4f} | ${dsp_block.get('savings_usd',0):.4f} | {dsp_block.get('long_context_batch_mode',False)} |")
    if has_pro and gmp_block:
        cost_rows.append(f"| {gm_pro_label} | {gmp_block.get('total_input_tokens',0):,} | {gmp_block.get('total_output_tokens',0):,} | ${gmp_block.get('cost_standard_usd',0):.4f} | ${gmp_block.get('cost_actual_usd',0):.4f} | ${gmp_block.get('savings_usd',0):.4f} | {gmp_block.get('long_context_batch_mode',False)} |")

    lines += [
        "| Model | Input Tokens | Output Tokens | Without Batch | With Batch | Savings | Batch? |",
        "|-------|:---:|:---:|:---:|:---:|:---:|:---:|",
        *cost_rows,
        f"| Gemini Pro (judge) | — | — | — | ${cost_summary['judge_gemini_pro']['estimated_cost_usd']:.4f} | — | no |",
        f"| **Total** | | | **${totals['cost_without_batch_usd']:.4f}** | **${totals['cost_with_batch_usd']:.4f}** | **${totals['batch_savings_usd']:.4f} ({totals['batch_savings_pct']}%)** | |",
        "",
        "> Pricing: doubleword.ai (DeepSeek) · ai.google.dev/pricing (Gemini)",
        "> Doubleword delayed batch (1h window): 50% off. Gemini 3.1 Pro Preview: ≤200k $1/$6, >200k $2/$9 per 1M tokens.",
        "",
        "---",
        "",
        "## Overall Verdict",
    ]

    ds_total_wins = ds_lc.get("wins", 0) + ds_tc.get("wins", 0)
    gm_total_wins = gm_lc.get("wins", 0) + gm_tc.get("wins", 0)
    overall_flash = (
        ds_flash_label if ds_total_wins > gm_total_wins else
        (gm_flash_label if gm_total_wins > ds_total_wins else "Tie")
    )

    lines += [
        f"| Dimension | {ds_flash_label} | {gm_flash_label} |",
        f"|-----------|:---:|:---:|",
        f"| Long Context Score | {ds_lc.get('total_score','?')}% | {gm_lc.get('total_score','?')}% |",
        f"| Tool Calling Score | {ds_tc.get('total_score','?')}% | {gm_tc.get('total_score','?')}% |",
        f"| Total Wins | {ds_total_wins} | {gm_total_wins} |",
        f"| Cost (batch) | ${ds_block.get('cost_actual_usd',0):.4f} | ${gm_block.get('cost_actual_usd',0):.4f} |",
        "",
        f"### **Flash winner: {overall_flash}**",
    ]

    if has_pro:
        dsp_total = dsp_lc.get("wins", 0) + dsp_tc.get("wins", 0)
        gmp_total = gmp_lc.get("wins", 0) + gmp_tc.get("wins", 0)
        overall_pro = (
            ds_pro_label if dsp_total > gmp_total else
            (gm_pro_label if gmp_total > dsp_total else "Tie")
        )
        lines += [
            "",
            f"| Dimension | {ds_pro_label} | {gm_pro_label} |",
            f"|-----------|:---:|:---:|",
            f"| Long Context Score | {dsp_lc.get('total_score','?')}% | {gmp_lc.get('total_score','?')}% |",
            f"| Tool Calling Score | {dsp_tc.get('total_score','?')}% | {gmp_tc.get('total_score','?')}% |",
            f"| Total Wins | {dsp_total} | {gmp_total} |",
            f"| Cost (batch) | ${dsp_block.get('cost_actual_usd',0):.4f} | ${gmp_block.get('cost_actual_usd',0):.4f} |",
            "",
            f"### **Pro winner: {overall_pro}**",
        ]

    lines += [
        "",
        "---",
        "*Results in `results/`. Flash: `long_context_results.json`, `tool_calling_results.json`. "
        "Pro: `long_context_results_pro.json`, `tool_calling_results_pro.json`.*",
    ]

    report = "\n".join(lines)
    path = os.path.join(results_dir, "final_report.md")
    with open(path, "w") as f:
        f.write(report)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DeepSeek vs Gemini medical evaluation")
    p.add_argument("--skip-long-context",  action="store_true", help="Skip Part 1 entirely (load from results/ if available)")
    p.add_argument("--skip-deepseek-lc",   action="store_true", help="Skip DeepSeek LC batch; load DS answers from saved results, re-run Gemini")
    p.add_argument("--skip-gemini-lc",     action="store_true", help="Skip Gemini LC; load GM answers from saved results, re-run DeepSeek")
    p.add_argument("--skip-tool-calling",  action="store_true", help="Skip Part 2 (load from results/ if available)")
    p.add_argument("--skip-judge",         action="store_true", help="Skip judge evaluation")
    p.add_argument("--skip-report",        action="store_true", help="Skip final report generation")
    p.add_argument("--include-pro",        action="store_true",
                   help="Also run Pro models (DEEPSEEK_PRO_MODEL + GEMINI_EVAL_PRO_MODEL) and add a 4-way comparison to the report")
    return p.parse_args()


def load_json_if_exists(path: str):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _responses_from_saved_lc(
    saved: dict,
    model_key: str,
    model_name: str,
) -> "Dict[int, List]":
    """
    Reconstruct per-tier ModelResponse lists from long_context_results.json.
    Used by --skip-deepseek-lc / --skip-gemini-lc to re-use one model's saved
    answers while the other model is re-evaluated.
    """
    from medical_texts import CONTEXT_TIERS
    from models import ModelResponse

    tier_responses: Dict[int, List] = {}
    for tier in CONTEXT_TIERS:
        label = f"{tier // 1000}k"
        tier_data = saved.get("tiers", {}).get(label, {})
        responses = []
        for q in tier_data.get("questions", []):
            r = q.get(model_key) or {}
            responses.append(ModelResponse(
                model_name=model_name,
                final_text=r.get("answer") or "",
                input_tokens=r.get("input_tokens", 0),
                output_tokens=r.get("output_tokens", 0),
                latency_s=float(r.get("latency_s") or 0.0),
                batch_mode=bool(r.get("batch_mode", False)),
                error=r.get("error"),
            ))
        tier_responses[tier] = responses
    return tier_responses


def main():
    args = parse_args()

    print("=" * 70)
    print("  DeepSeek-V4-Flash vs Gemini 3 Flash — Medical Evaluation")
    print("=" * 70)

    try:
        config = load_config()
    except EnvironmentError as exc:
        print(f"\nConfiguration error:\n{exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nModels:")
    print(f"  DeepSeek : {config.deepseek_model}  (via {config.doubleword_base_url})")
    print(f"  Flash    : {config.gemini_flash_model}")
    print(f"  Judge    : {config.gemini_pro_model}")
    print(f"\nBatch mode:")
    print(f"  Doubleword delayed (1h window) : {config.use_doubleword_batch}")
    print(f"  Gemini batch API               : {config.use_gemini_batch}")

    os.makedirs(config.results_dir, exist_ok=True)

    # Initialise models
    deepseek     = DeepSeekModel(config.doubleword_api_key, config.doubleword_base_url, config.deepseek_model)
    gemini_flash = GeminiModel(config.gemini_api_key, config.gemini_flash_model)
    gemini_pro   = GeminiModel(config.gemini_api_key, config.gemini_pro_model)

    eval_start = time.time()
    lc_path = os.path.join(config.results_dir, "long_context_results.json")
    tc_path = os.path.join(config.results_dir, "tool_calling_results.json")

    # ── Async flow for long context + tool calling ────────────────────────────
    #
    # Timeline (batch enabled, full run):
    #   T=0   Build/load all 4 tier corpora (cached after first run)
    #   T=0   Submit all 4×5=20 DeepSeek LC questions in ONE batch (non-blocking)
    #   T=0   Run all 4 Gemini tier queries sequentially
    #   T=~   Run tool calling for BOTH models (sync, fills time while batch queues)
    #   T=~   Collect DeepSeek batch (poll — likely done by now)
    #   T=~   Save results, run judge, write report
    #
    # Partial re-run flags (useful when one model errors):
    #   --skip-long-context  : skip Part 1 entirely, load saved results
    #   --skip-deepseek-lc   : load saved DS answers, re-run Gemini only
    #   --skip-gemini-lc     : load saved Gemini answers, re-run DeepSeek only

    lc_results = None
    if args.skip_long_context:
        lc_results = load_json_if_exists(lc_path)
        if lc_results:
            print(f"\n[skip] Loaded long context results from {lc_path}")
        else:
            print("\n[skip] --skip-long-context set but no saved results found; running full eval...")
            args.skip_long_context = False

    ds_batch_id = None
    ds_tier_responses = None
    gm_tier_responses = None
    tier_corpora = None

    if not args.skip_long_context:
        print("\n" + "=" * 70)
        print("PART 1: LONG CONTEXT — NEEDLE-IN-A-HAYSTACK (4 tiers)")
        print("=" * 70)
        tier_corpora = prepare_all_tier_corpora(config)

        # ── DeepSeek LC ───────────────────────────────────────────────────────
        if args.skip_deepseek_lc:
            saved = load_json_if_exists(lc_path)
            if saved and "tiers" in saved:
                ds_tier_responses = _responses_from_saved_lc(
                    saved, "deepseek", config.deepseek_model
                )
                print(f"\n[skip-deepseek-lc] Loaded saved DeepSeek LC answers from {lc_path}")
            else:
                print("\n[skip-deepseek-lc] No saved multi-tier results found; running DeepSeek LC...")
                args.skip_deepseek_lc = False

        if not args.skip_deepseek_lc:
            ds_batch_id = submit_all_tiers_batch(
                config, deepseek, tier_corpora, config.results_dir
            )

        # ── Gemini LC (runs while DS batch queues) ────────────────────────────
        if args.skip_gemini_lc:
            saved = load_json_if_exists(lc_path)
            if saved and "tiers" in saved:
                gm_tier_responses = _responses_from_saved_lc(
                    saved, "gemini_flash", config.gemini_flash_model
                )
                print(f"\n[skip-gemini-lc] Loaded saved Gemini LC answers from {lc_path}")
            else:
                print("\n[skip-gemini-lc] No saved multi-tier results found; running Gemini LC...")
                args.skip_gemini_lc = False

        if not args.skip_gemini_lc:
            gm_tier_responses = run_all_gemini_tiers(config, gemini_flash, tier_corpora)

    # ── Part 2: Tool calling (runs while DeepSeek batch is pending) ───────────
    if args.skip_tool_calling:
        tc_results = load_json_if_exists(tc_path)
        if tc_results:
            print(f"\n[skip] Loaded tool calling results from {tc_path}")
        else:
            print("\n[skip] --skip-tool-calling set but no saved results found; running eval...")
            tc_results = run_tool_calling_eval(config, deepseek, gemini_flash, config.results_dir)
    else:
        tc_results = run_tool_calling_eval(config, deepseek, gemini_flash, config.results_dir)

    # ── Collect DeepSeek batch / merge and save ───────────────────────────────
    if not args.skip_long_context:
        if not args.skip_deepseek_lc:
            ds_tier_responses = collect_all_tiers_batch(
                config, deepseek, tier_corpora, ds_batch_id, config.results_dir
            )
        lc_results = save_all_tier_results(
            config, tier_corpora,
            ds_tier_responses, gm_tier_responses,
            config.results_dir,
        )

    # ── Optional Pro model run ────────────────────────────────────────────────
    lc_results_pro = tc_results_pro = lc_eval_pro = tc_eval_pro = None
    lc_path_pro = os.path.join(config.results_dir, "long_context_results_pro.json")
    tc_path_pro = os.path.join(config.results_dir, "tool_calling_results_pro.json")

    run_pro = args.include_pro and (config.deepseek_pro_model or config.gemini_eval_pro_model)
    if args.include_pro and not run_pro:
        print("\n[pro] --include-pro set but neither DEEPSEEK_PRO_MODEL nor GEMINI_EVAL_PRO_MODEL is configured. Skipping.")

    if run_pro:
        print("\n" + "=" * 70)
        print("PRO MODEL COMPARISON")
        print("=" * 70)

        deepseek_pro   = DeepSeekModel(config.doubleword_api_key, config.doubleword_base_url,
                                       config.deepseek_pro_model) if config.deepseek_pro_model else deepseek
        gemini_eval_pro = GeminiModel(config.gemini_api_key,
                                      config.gemini_eval_pro_model) if config.gemini_eval_pro_model else gemini_flash

        print(f"  DeepSeek Pro : {config.deepseek_pro_model or '(same as flash)'}")
        print(f"  Gemini Pro   : {config.gemini_eval_pro_model or '(same as flash)'}")

        lc_results_pro = load_json_if_exists(lc_path_pro)
        if lc_results_pro:
            print(f"\n[skip] Loaded pro LC results from {lc_path_pro}")
        else:
            # Ensure corpora are built (may already be done from flash run)
            if tier_corpora is None:
                tier_corpora = prepare_all_tier_corpora(config)

            # Submit DeepSeek Pro batch (24h window — same API, completion_window="1h")
            ds_pro_batch_id = submit_all_tiers_batch(
                config, deepseek_pro, tier_corpora, config.results_dir,
                q_prefix="lc-pro", batch_id_filename="deepseek_pro_batch_id.txt",
            )

            # Run Gemini Pro tiers (use pro batch setting)
            gm_pro_tier_responses = run_all_gemini_tiers(
                config, gemini_eval_pro, tier_corpora,
                use_batch=config.use_gemini_pro_batch,
            )

            # Collect DeepSeek Pro batch
            ds_pro_tier_responses = collect_all_tiers_batch(
                config, deepseek_pro, tier_corpora, ds_pro_batch_id,
                config.results_dir, q_prefix="lc-pro",
            )

            lc_results_pro = save_all_tier_results(
                config, tier_corpora,
                ds_pro_tier_responses, gm_pro_tier_responses,
                config.results_dir,
                save_suffix="_pro",
                deepseek_label=config.deepseek_pro_model,
                gemini_label=config.gemini_eval_pro_model,
            )

        if args.skip_tool_calling:
            tc_results_pro = load_json_if_exists(tc_path_pro)
            if tc_results_pro:
                print(f"\n[skip] Loaded pro TC results from {tc_path_pro}")
        if tc_results_pro is None:
            tc_results_pro = run_tool_calling_eval(
                config, deepseek_pro, gemini_eval_pro, config.results_dir,
                save_suffix="_pro",
                deepseek_label=config.deepseek_pro_model,
                gemini_label=config.gemini_eval_pro_model,
            )


    # ── Judge ─────────────────────────────────────────────────────────────────
    lc_eval_path = os.path.join(config.results_dir, "long_context_evaluation.json")
    tc_eval_path = os.path.join(config.results_dir, "tool_calling_evaluation.json")

    if args.skip_judge:
        lc_eval = load_json_if_exists(lc_eval_path) or {}
        tc_eval = load_json_if_exists(tc_eval_path) or {}
        if run_pro:
            lc_eval_pro = load_json_if_exists(os.path.join(config.results_dir, "long_context_evaluation_pro.json")) or {}
            tc_eval_pro = load_json_if_exists(os.path.join(config.results_dir, "tool_calling_evaluation_pro.json")) or {}
        print(f"\n[skip] Loaded judge evaluations from {config.results_dir}/")
    else:
        lc_eval = evaluate_long_context(config, gemini_pro, lc_results, config.results_dir)
        tc_eval = evaluate_tool_calling(config, gemini_pro, tc_results, config.results_dir)
        if run_pro and lc_results_pro:
            lc_eval_pro = evaluate_long_context(config, gemini_pro, lc_results_pro, config.results_dir, save_suffix="_pro")
        if run_pro and tc_results_pro:
            tc_eval_pro = evaluate_tool_calling(config, gemini_pro, tc_results_pro, config.results_dir, save_suffix="_pro")

    # ── Cost summary ──────────────────────────────────────────────────────────
    cost_summary = compute_cost_summary(
        config, lc_results, tc_results, lc_eval, tc_eval,
        lc_results_pro, tc_results_pro,
    )
    cost_path = os.path.join(config.results_dir, "cost_summary.json")
    with open(cost_path, "w") as f:
        json.dump(cost_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("COST SUMMARY")
    print("=" * 70)
    totals = cost_summary["totals"]
    for key in ("deepseek_flash", "gemini_flash", "deepseek_pro", "gemini_eval_pro"):
        c = cost_summary.get(key)
        if c and c.get("total_input_tokens", 0):
            print(f"  {c['model']:<45} {c['total_input_tokens']:>10,} in  {c['total_output_tokens']:>8,} out"
                  f"  → ${c['cost_standard_usd']:.4f} std | ${c['cost_actual_usd']:.4f} batch"
                  f"  (saves ${c['savings_usd']:.4f})")
    print(f"  Gemini Pro (judge) : ~${cost_summary['judge_gemini_pro']['estimated_cost_usd']:.4f}")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  TOTAL without batch : ${totals['cost_without_batch_usd']:.4f}")
    print(f"  TOTAL with batch    : ${totals['cost_with_batch_usd']:.4f}  "
          f"(saves ${totals['batch_savings_usd']:.4f} / {totals['batch_savings_pct']}%)")

    # ── Final report ──────────────────────────────────────────────────────────
    if not args.skip_report:
        report_path = write_report(
            config, lc_results, tc_results, lc_eval, tc_eval, cost_summary, config.results_dir,
            lc_results_pro=lc_results_pro, tc_results_pro=tc_results_pro,
            lc_eval_pro=lc_eval_pro, tc_eval_pro=tc_eval_pro,
        )
        print(f"\nFinal report: {report_path}")

    elapsed = time.time() - eval_start
    print(f"\nTotal wall-clock time: {elapsed/60:.1f} min")
    print("\nAll results saved to:", config.results_dir)
    print("  long_context_results.json      (4 tiers × 5 questions)")
    print("  tool_calling_results.json      (5 scenarios)")
    print("  long_context_evaluation.json   (judge scores per tier)")
    print("  tool_calling_evaluation.json   (judge scores per scenario)")
    print("  cost_summary.json")
    print("  final_report.md")
    print("  corpus_cache_200k.txt  corpus_cache_400k.txt  ...")
    print("  (delete corpus_cache_*.txt to force a fresh HuggingFace download)")


if __name__ == "__main__":
    main()
