#!/usr/bin/env python3
"""Measure token cost of CoPilot system prompt + tool schemas.

Outputs a detailed breakdown and dumps the full prompt to a text file
for before/after comparison.

Usage:
    cd autogpt_platform/backend
    poetry run python scripts/measure_copilot_tokens.py [--output before.txt]
"""

import argparse
import json
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.copilot.prompting import _SHARED_TOOL_NOTES, get_sdk_supplement
from backend.copilot.tools import TOOL_REGISTRY


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return len(text) // 4


def measure_tool_schema(name: str, tool) -> dict:
    """Measure a single tool's schema contribution."""
    schema = tool.as_openai_tool()
    schema_json = json.dumps(schema, indent=2)
    desc = schema["function"].get("description", "")
    params = schema["function"].get("parameters", {})
    params_json = json.dumps(params, indent=2)

    return {
        "name": name,
        "description_chars": len(desc),
        "description_tokens": estimate_tokens(desc),
        "params_chars": len(params_json),
        "params_tokens": estimate_tokens(params_json),
        "total_chars": len(schema_json),
        "total_tokens": estimate_tokens(schema_json),
        "schema_json": schema_json,
        "description": desc,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure CoPilot token costs")
    parser.add_argument(
        "--output", "-o", default=None, help="Output file for full prompt dump"
    )
    args = parser.parse_args()

    # System prompt (the default one from service.py)
    system_prompt = """You are an AI automation assistant helping users build and run automations.

Here is everything you know about the current user from previous interactions:

<users_information>
(user context would go here)
</users_information>

Your goal is to help users automate tasks by:
- Understanding their needs and business context
- Building and running working automations
- Delivering tangible value through action, not just explanation

Be concise, proactive, and action-oriented. Bias toward showing working solutions over lengthy explanations."""

    sdk_supplement = get_sdk_supplement(use_e2b=False, cwd="/tmp/copilot-session123")

    # Measure system prompt
    full_system = system_prompt + sdk_supplement
    sys_tokens = estimate_tokens(full_system)

    print("=" * 70)
    print("COPILOT TOKEN BREAKDOWN")
    print("=" * 70)
    print(
        f"\nSystem Prompt: {len(system_prompt)} chars, ~{estimate_tokens(system_prompt)} tokens"
    )
    print(
        f"SDK Supplement: {len(sdk_supplement)} chars, ~{estimate_tokens(sdk_supplement)} tokens"
    )
    print(
        f"  - Shared Tool Notes: {len(_SHARED_TOOL_NOTES)} chars, ~{estimate_tokens(_SHARED_TOOL_NOTES)} tokens"
    )
    print(f"Full System: {len(full_system)} chars, ~{sys_tokens} tokens")

    # Measure each tool
    print(f"\n{'=' * 70}")
    print("TOOL SCHEMAS")
    print(f"{'=' * 70}")

    tools = []
    for name, tool in sorted(TOOL_REGISTRY.items()):
        if not tool.is_available:
            continue
        info = measure_tool_schema(name, tool)
        tools.append(info)

    # Sort by total tokens descending
    tools.sort(key=lambda x: x["total_tokens"], reverse=True)

    total_tool_tokens = 0
    total_desc_tokens = 0
    total_params_tokens = 0

    print(
        f"\n{'Tool':<30} {'Desc Tok':>10} {'Param Tok':>10} {'Total Tok':>10} {'Desc Chars':>10}"
    )
    print("-" * 72)
    for t in tools:
        print(
            f"{t['name']:<30} {t['description_tokens']:>10} "
            f"{t['params_tokens']:>10} {t['total_tokens']:>10} "
            f"{t['description_chars']:>10}"
        )
        total_tool_tokens += t["total_tokens"]
        total_desc_tokens += t["description_tokens"]
        total_params_tokens += t["params_tokens"]

    print("-" * 72)
    print(
        f"{'TOTAL (' + str(len(tools)) + ' tools)':<30} {total_desc_tokens:>10} "
        f"{total_params_tokens:>10} {total_tool_tokens:>10}"
    )

    grand_total = sys_tokens + total_tool_tokens
    print(f"\n{'=' * 70}")
    print("GRAND TOTAL")
    print(f"{'=' * 70}")
    print(
        f"System Prompt:  ~{sys_tokens:>6} tokens ({sys_tokens * 100 // grand_total}%)"
    )
    print(
        f"Tool Schemas:   ~{total_tool_tokens:>6} tokens ({total_tool_tokens * 100 // grand_total}%)"
    )
    print(f"  Descriptions: ~{total_desc_tokens:>6} tokens")
    print(f"  Parameters:   ~{total_params_tokens:>6} tokens")
    print(f"GRAND TOTAL:    ~{grand_total:>6} tokens")

    # Dump to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("SYSTEM PROMPT\n")
            f.write("=" * 70 + "\n")
            f.write(full_system)
            f.write("\n\n")
            f.write("=" * 70 + "\n")
            f.write(f"TOOL SCHEMAS ({len(tools)} tools)\n")
            f.write("=" * 70 + "\n\n")
            for t in tools:
                f.write(f"--- {t['name']} ---\n")
                f.write(
                    f"Description ({t['description_chars']} chars, ~{t['description_tokens']} tokens):\n"
                )
                f.write(t["description"] + "\n\n")
                f.write(
                    f"Schema ({t['total_chars']} chars, ~{t['total_tokens']} tokens):\n"
                )
                f.write(t["schema_json"] + "\n\n")
        print(f"\nFull dump written to: {output_path}")


if __name__ == "__main__":
    main()
