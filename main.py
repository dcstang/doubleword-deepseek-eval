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
) -> dict:
    """Aggregate token usage and compute estimated costs with/without batch discount."""

    def tally(results_part: dict, model_key: str):
        inp = out = 0
        batch = False
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

    ds_total_in  = ds_lc_in  + ds_tc_in
    ds_total_out = ds_lc_out + ds_tc_out
    gm_total_in  = gm_lc_in  + gm_tc_in
    gm_total_out = gm_lc_out + gm_tc_out

    p_ds = config.deepseek_pricing
    p_gm = config.gemini_flash_pricing
    p_pro = config.gemini_pro_pricing

    # DeepSeek: long-context questions in batch, tool-calling sync
    ds_cost_standard = p_ds.cost(ds_total_in, ds_total_out, batch=False)
    ds_lc_cost_batch = p_ds.cost(ds_lc_in, ds_lc_out, batch=True) if ds_lc_batch else p_ds.cost(ds_lc_in, ds_lc_out, batch=False)
    ds_tc_cost       = p_ds.cost(ds_tc_in, ds_tc_out, batch=False)
    ds_cost_actual   = ds_lc_cost_batch + ds_tc_cost

    # Gemini Flash: long-context in batch if enabled, tool-calling sync
    gm_cost_standard = p_gm.cost(gm_total_in, gm_total_out, batch=False)
    gm_lc_cost_batch = p_gm.cost(gm_lc_in, gm_lc_out, batch=True) if gm_lc_batch else p_gm.cost(gm_lc_in, gm_lc_out, batch=False)
    gm_tc_cost       = p_gm.cost(gm_tc_in, gm_tc_out, batch=False)
    gm_cost_actual   = gm_lc_cost_batch + gm_tc_cost

    # Judge (Gemini Pro — rough estimate: assume ~2k in / 0.5k out per judgment, 10 judgments)
    judge_in_est  = 10 * 2000
    judge_out_est = 10 * 500
    judge_cost    = p_pro.cost(judge_in_est, judge_out_est)

    total_standard = ds_cost_standard + gm_cost_standard + judge_cost
    total_actual   = ds_cost_actual   + gm_cost_actual   + judge_cost
    savings        = total_standard - total_actual

    return {
        "deepseek": {
            "model": config.deepseek_model,
            "total_input_tokens": ds_total_in,
            "total_output_tokens": ds_total_out,
            "long_context_batch_mode": ds_lc_batch,
            "cost_standard_usd": round(ds_cost_standard, 4),
            "cost_actual_usd": round(ds_cost_actual, 4),
            "savings_usd": round(ds_cost_standard - ds_cost_actual, 4),
            "pricing": {
                "input_per_million": p_ds.input_per_million,
                "output_per_million": p_ds.output_per_million,
                "batch_discount_pct": int(p_ds.batch_discount * 100),
            },
        },
        "gemini_flash": {
            "model": config.gemini_flash_model,
            "total_input_tokens": gm_total_in,
            "total_output_tokens": gm_total_out,
            "long_context_batch_mode": gm_lc_batch,
            "cost_standard_usd": round(gm_cost_standard, 4),
            "cost_actual_usd": round(gm_cost_actual, 4),
            "savings_usd": round(gm_cost_standard - gm_cost_actual, 4),
            "pricing": {
                "input_per_million": p_gm.input_per_million,
                "output_per_million": p_gm.output_per_million,
                "batch_discount_pct": int(p_gm.batch_discount * 100),
            },
        },
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
) -> str:
    # Multi-tier uses overall_summary; legacy single-tier uses summary
    lc_overall = lc_eval.get("overall_summary", lc_eval.get("summary", {}))
    tc_sum = tc_eval.get("summary", {})
    ds_lc = lc_overall.get("deepseek", {})
    gm_lc = lc_overall.get("gemini_flash", {})
    ds_tc = tc_sum.get("deepseek", {})
    gm_tc = tc_sum.get("gemini_flash", {})

    ds_cost = cost_summary["deepseek"]
    gm_cost = cost_summary["gemini_flash"]
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

    lines += [
        "",
        "### Tool Calling Summary",
        f"| Model | Avg Tool Sel | Avg Params | Avg Clinical | Score | Wins |",
        f"|-------|:---:|:---:|:---:|:---:|:---:|",
        f"| DeepSeek-V4-Flash | {ds_tc.get('avg_tool_selection','?')}/3 | {ds_tc.get('avg_parameter_accuracy','?')}/3 | {ds_tc.get('avg_clinical_reasoning','?')}/4 | {ds_tc.get('total_score','?')}% | {ds_tc.get('wins','?')} |",
        f"| Gemini 3 Flash    | {gm_tc.get('avg_tool_selection','?')}/3 | {gm_tc.get('avg_parameter_accuracy','?')}/3 | {gm_tc.get('avg_clinical_reasoning','?')}/4 | {gm_tc.get('total_score','?')}% | {gm_tc.get('wins','?')} |",
        "",
        f"**Tool calling winner: {tc_sum.get('overall_winner', '?').upper()}**",
        "",
        "---",
        "",
        "## Cost Comparison",
        "",
        "### Token Usage",
        f"| Model | Input Tokens | Output Tokens |",
        f"|-------|:---:|:---:|",
        f"| {config.deepseek_model} | {ds_cost['total_input_tokens']:,} | {ds_cost['total_output_tokens']:,} |",
        f"| {config.gemini_flash_model} | {gm_cost['total_input_tokens']:,} | {gm_cost['total_output_tokens']:,} |",
        "",
        "### Estimated Cost (USD)",
        f"| Model | Without Batch | With Batch/Delayed | Savings | Batch Mode Used |",
        f"|-------|:---:|:---:|:---:|:---:|",
        f"| DeepSeek-V4-Flash | ${ds_cost['cost_standard_usd']:.4f} | ${ds_cost['cost_actual_usd']:.4f} | ${ds_cost['savings_usd']:.4f} | {ds_cost['long_context_batch_mode']} |",
        f"| Gemini 3 Flash    | ${gm_cost['cost_standard_usd']:.4f} | ${gm_cost['cost_actual_usd']:.4f} | ${gm_cost['savings_usd']:.4f} | {gm_cost['long_context_batch_mode']} |",
        f"| Gemini 3 Pro (judge) | — | ${cost_summary['judge_gemini_pro']['estimated_cost_usd']:.4f} | — | no |",
        f"| **Total** | **${totals['cost_without_batch_usd']:.4f}** | **${totals['cost_with_batch_usd']:.4f}** | **${totals['batch_savings_usd']:.4f} ({totals['batch_savings_pct']}%)** | |",
        "",
        "> Pricing estimates — verify current rates at doubleword.ai and ai.google.dev/pricing.",
        "> Doubleword delayed batch (1h window) = 50% discount on long-context queries.",
        "",
        "### Pricing Config Used",
        f"| Model | Input $/M | Output $/M | Batch Discount |",
        f"|-------|:---:|:---:|:---:|",
        f"| DeepSeek (Doubleword) | ${ds_cost['pricing']['input_per_million']} | ${ds_cost['pricing']['output_per_million']} | {ds_cost['pricing']['batch_discount_pct']}% |",
        f"| Gemini 3 Flash | ${gm_cost['pricing']['input_per_million']} | ${gm_cost['pricing']['output_per_million']} | {gm_cost['pricing']['batch_discount_pct']}% |",
        "",
        "---",
        "",
        "## Overall Verdict",
    ]

    ds_total_wins = ds_lc.get("wins", 0) + ds_tc.get("wins", 0)
    gm_total_wins = gm_lc.get("wins", 0) + gm_tc.get("wins", 0)
    overall = (
        "DeepSeek-V4-Flash" if ds_total_wins > gm_total_wins else
        ("Gemini 3 Flash" if gm_total_wins > ds_total_wins else "Tie")
    )

    lines += [
        f"| Dimension | DeepSeek-V4-Flash | Gemini 3 Flash |",
        f"|-----------|:-----------------:|:--------------:|",
        f"| Long Context Score | {ds_lc.get('total_score', '?')}% | {gm_lc.get('total_score', '?')}% |",
        f"| Tool Calling Score | {ds_tc.get('total_score', '?')}% | {gm_tc.get('total_score', '?')}% |",
        f"| Total Wins | {ds_total_wins} | {gm_total_wins} |",
        f"| Estimated Cost (batch) | ${ds_cost['cost_actual_usd']:.4f} | ${gm_cost['cost_actual_usd']:.4f} |",
        "",
        f"### **Overall winner: {overall}**",
        "",
        "---",
        "*Results are in `results/` directory. Raw model outputs: `long_context_results.json`, `tool_calling_results.json`. Judge scores: `long_context_evaluation.json`, `tool_calling_evaluation.json`.*",
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
    p.add_argument("--skip-long-context", action="store_true", help="Skip Part 1 (load from results/ if available)")
    p.add_argument("--skip-tool-calling", action="store_true", help="Skip Part 2 (load from results/ if available)")
    p.add_argument("--skip-judge", action="store_true", help="Skip judge evaluation")
    p.add_argument("--skip-report", action="store_true", help="Skip final report generation")
    return p.parse_args()


def load_json_if_exists(path: str):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


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
    # Timeline (batch enabled):
    #   T=0   Build/load all 4 tier corpora (cached after first run)
    #   T=0   Submit all 4×5=20 DeepSeek LC questions in ONE batch (non-blocking)
    #   T=0   Run all 4 Gemini tier queries sequentially
    #   T=~   Run tool calling for BOTH models (sync, fills time while batch queues)
    #   T=~   Collect DeepSeek batch (poll — likely done by now)
    #   T=~   Save results, run judge, write report
    #
    # With --skip-long-context both LC steps are skipped and saved results loaded.

    if args.skip_long_context:
        lc_results = load_json_if_exists(lc_path)
        if lc_results:
            print(f"\n[skip] Loaded long context results from {lc_path}")
        else:
            print("\n[skip] --skip-long-context set but no saved results found; running eval...")
            args.skip_long_context = False  # fall through to full eval below

    ds_batch_id = None
    gm_tier_responses = None
    tier_corpora = None

    if not args.skip_long_context:
        print("\n" + "=" * 70)
        print("PART 1: LONG CONTEXT — NEEDLE-IN-A-HAYSTACK (4 tiers)")
        print("=" * 70)
        tier_corpora = prepare_all_tier_corpora(config)

        # Fire all-tier DeepSeek batch (non-blocking)
        ds_batch_id = submit_all_tiers_batch(
            config, deepseek, tier_corpora, config.results_dir
        )

        # Run all Gemini tiers while batch queues
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

    # ── Collect DeepSeek batch now (it's had time to process) ────────────────
    if not args.skip_long_context:
        ds_tier_responses = collect_all_tiers_batch(
            config, deepseek, tier_corpora, ds_batch_id, config.results_dir
        )
        lc_results = save_all_tier_results(
            config, tier_corpora,
            ds_tier_responses, gm_tier_responses,
            config.results_dir,
        )

    # ── Judge ─────────────────────────────────────────────────────────────────
    lc_eval_path = os.path.join(config.results_dir, "long_context_evaluation.json")
    tc_eval_path = os.path.join(config.results_dir, "tool_calling_evaluation.json")

    if args.skip_judge:
        lc_eval = load_json_if_exists(lc_eval_path) or {}
        tc_eval = load_json_if_exists(tc_eval_path) or {}
        print(f"\n[skip] Loaded judge evaluations from {config.results_dir}/")
    else:
        lc_eval = evaluate_long_context(config, gemini_pro, lc_results, config.results_dir)
        tc_eval = evaluate_tool_calling(config, gemini_pro, tc_results, config.results_dir)

    # ── Cost summary ──────────────────────────────────────────────────────────
    cost_summary = compute_cost_summary(config, lc_results, tc_results, lc_eval, tc_eval)
    cost_path = os.path.join(config.results_dir, "cost_summary.json")
    with open(cost_path, "w") as f:
        json.dump(cost_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("COST SUMMARY")
    print("=" * 70)
    ds_c = cost_summary["deepseek"]
    gm_c = cost_summary["gemini_flash"]
    totals = cost_summary["totals"]
    print(f"  DeepSeek-V4-Flash : {ds_c['total_input_tokens']:,} in / {ds_c['total_output_tokens']:,} out "
          f"→ ${ds_c['cost_standard_usd']:.4f} standard | ${ds_c['cost_actual_usd']:.4f} batch "
          f"(saves ${ds_c['savings_usd']:.4f})")
    print(f"  Gemini 3 Flash    : {gm_c['total_input_tokens']:,} in / {gm_c['total_output_tokens']:,} out "
          f"→ ${gm_c['cost_standard_usd']:.4f} standard | ${gm_c['cost_actual_usd']:.4f} batch "
          f"(saves ${gm_c['savings_usd']:.4f})")
    print(f"  Gemini 3 Pro (judge) : ~${cost_summary['judge_gemini_pro']['estimated_cost_usd']:.4f}")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  TOTAL without batch : ${totals['cost_without_batch_usd']:.4f}")
    print(f"  TOTAL with batch    : ${totals['cost_with_batch_usd']:.4f}  "
          f"(saves ${totals['batch_savings_usd']:.4f} / {totals['batch_savings_pct']}%)")

    # ── Final report ──────────────────────────────────────────────────────────
    if not args.skip_report:
        report_path = write_report(
            config, lc_results, tc_results, lc_eval, tc_eval, cost_summary, config.results_dir
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
