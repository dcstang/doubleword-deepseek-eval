"""
Part 2: Tool-calling evaluation.

Presents 5 medical scenarios to both models with a set of medical and general tools.
Captures which tools were called, with what arguments, and what the final clinical
recommendation was.
"""

import json
import os

from config import Config
from models import DeepSeekModel, GeminiModel, ModelResponse
from scenarios import TOOL_CALLING_SCENARIOS


SYSTEM_PROMPT = (
    "You are an expert clinical decision support assistant. "
    "You have access to medical scoring tools and reference databases. "
    "For each patient scenario: use the available tools to calculate relevant scores, "
    "then synthesise the results into a concise clinical assessment and management recommendation. "
    "Always call the tools with the exact parameter values from the case description."
)


def run_tool_calling_eval(
    config: Config,
    deepseek: DeepSeekModel,
    gemini_flash: GeminiModel,
    results_dir: str,
) -> dict:
    print("\n" + "=" * 70)
    print("PART 2: TOOL CALLING EVALUATION")
    print("=" * 70)

    results = {
        "part": "tool_calling",
        "deepseek_model": config.deepseek_model,
        "gemini_flash_model": config.gemini_flash_model,
        "scenarios": [],
    }

    for scenario in TOOL_CALLING_SCENARIOS:
        s_id = scenario["id"]
        title = scenario["title"]
        prompt = scenario["prompt"]
        available_tools = scenario["available_tools"]
        expected_tools = scenario["expected_tools"]
        reference = scenario["reference_answer"]

        print(f"\n  Scenario {s_id}: {title}")
        print(f"  Available tools: {available_tools}")
        print(f"  Expected tools:  {expected_tools}")

        s_result = {
            "scenario_id": s_id,
            "title": title,
            "available_tools": available_tools,
            "expected_tools": expected_tools,
            "reference_answer": reference,
            "deepseek": None,
            "gemini_flash": None,
        }

        # DeepSeek
        print(f"\n    Querying {config.deepseek_model}...")
        ds_resp: ModelResponse = deepseek.chat_with_tools(
            prompt=prompt,
            tool_names=available_tools,
            system=SYSTEM_PROMPT,
            max_tokens=1024,
        )
        s_result["deepseek"] = {
            "final_text": ds_resp.final_text,
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments, "result": tc.result}
                for tc in ds_resp.tool_calls
            ],
            "tools_used": [tc.name for tc in ds_resp.tool_calls],
            "latency_s": ds_resp.latency_s,
            "input_tokens": ds_resp.input_tokens,
            "output_tokens": ds_resp.output_tokens,
            "error": ds_resp.error,
        }
        if ds_resp.error:
            print(f"    ERROR: {ds_resp.error}")
        else:
            tools_used = [tc.name for tc in ds_resp.tool_calls]
            print(f"    Tools called: {tools_used}")
            print(f"    Response ({ds_resp.latency_s:.1f}s): {ds_resp.final_text[:150]}...")

        # Gemini Flash
        print(f"\n    Querying {config.gemini_flash_model}...")
        gm_resp: ModelResponse = gemini_flash.chat_with_tools(
            prompt=prompt,
            tool_names=available_tools,
            system=SYSTEM_PROMPT,
            max_tokens=1024,
        )
        s_result["gemini_flash"] = {
            "final_text": gm_resp.final_text,
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments, "result": tc.result}
                for tc in gm_resp.tool_calls
            ],
            "tools_used": [tc.name for tc in gm_resp.tool_calls],
            "latency_s": gm_resp.latency_s,
            "input_tokens": gm_resp.input_tokens,
            "output_tokens": gm_resp.output_tokens,
            "error": gm_resp.error,
        }
        if gm_resp.error:
            print(f"    ERROR: {gm_resp.error}")
        else:
            tools_used = [tc.name for tc in gm_resp.tool_calls]
            print(f"    Tools called: {tools_used}")
            print(f"    Response ({gm_resp.latency_s:.1f}s): {gm_resp.final_text[:150]}...")

        results["scenarios"].append(s_result)

    # Save intermediate results
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "tool_calling_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTool calling results saved to {out_path}")

    return results
