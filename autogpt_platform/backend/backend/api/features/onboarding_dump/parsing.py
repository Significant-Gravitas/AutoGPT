"""Shared LLM-output parsing for the brain-dump pipeline."""

import json
import re


def parse_response_json(content: str) -> dict | None:
    """Parse the model's JSON, tolerating markdown fences and preamble.

    Anthropic models have no OpenAI-style JSON mode, so the contract is
    prompt-level ("return ONLY valid JSON") and the parser forgives the
    two ways that commonly bends: a ```json fence around the object, or
    stray prose before/after it.
    """
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None
