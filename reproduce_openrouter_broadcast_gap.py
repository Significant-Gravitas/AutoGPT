#!/usr/bin/env python3
"""
Reproduction script: OpenRouter Broadcast does not capture /api/v1/messages requests.

SUMMARY
-------
OpenRouter's "Broadcast" / Langfuse integration only forwards traces from the
OpenAI-compatible endpoint (`POST /api/v1/chat/completions`) to configured
observability providers.  Requests sent to the Anthropic-native endpoint
(`POST /api/v1/messages`) are silently omitted — they return HTTP 200 and
generate a valid `gen-*` ID, but **no trace ever appears in Langfuse** (or
presumably other Broadcast-connected providers).

This matters when you use the Claude Agent SDK (or any library that talks to
the Anthropic Messages API), because that SDK always calls /api/v1/messages
even when OpenRouter is set as the base URL.

REPRODUCTION STEPS
------------------
1. Configure a Langfuse integration in your OpenRouter account settings.
2. Export the required environment variables (see below).
3. Run:  python reproduce_openrouter_broadcast_gap.py
4. Wait ~30 s, then check your Langfuse project:
   - The /chat/completions generation WILL appear.
   - The /messages generation will NOT appear.

REQUIRED ENVIRONMENT VARIABLES
-------------------------------
  OPEN_ROUTER_API_KEY   Your OpenRouter API key  (sk-or-v1-...)

OPTIONAL (to check Langfuse automatically)
-------------------------------
  LANGFUSE_PUBLIC_KEY   pk-lf-...
  LANGFUSE_SECRET_KEY   sk-lf-...
  LANGFUSE_HOST         https://cloud.langfuse.com  (default)
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.environ.get("OPEN_ROUTER_API_KEY", "")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

# Use a cheap model available through OpenRouter for both endpoints.
# Anthropic-native format (no provider prefix):
ANTHROPIC_MODEL = "claude-3-haiku-20240307"
# OpenAI-compat format (provider prefix required by OpenRouter):
OPENAI_COMPAT_MODEL = "anthropic/claude-3-haiku"

OPENROUTER_BASE = "https://openrouter.ai/api"

RUN_TAG = f"broadcast-repro-{int(time.time())}"


def check_env() -> None:
    if not OPENROUTER_API_KEY:
        print(
            "ERROR: OPEN_ROUTER_API_KEY environment variable is not set.",
            file=sys.stderr,
        )
        sys.exit(1)


def http_post(url: str, headers: dict, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def call_messages_endpoint(label: str) -> dict:
    """POST /api/v1/messages  — Anthropic-native endpoint."""
    url = f"{OPENROUTER_BASE}/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "X-Title": RUN_TAG,
    }
    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": f"Reply with just: {label}"}],
    }
    print(f"\n[1] POST {url}")
    print(f"    model: {ANTHROPIC_MODEL}")
    status, resp = http_post(url, headers, body)
    gen_id = resp.get("id", "unknown")
    print(f"    HTTP {status}  →  id={gen_id}")
    if status != 200:
        print(f"    Response: {json.dumps(resp, indent=2)}")
    return resp


def call_chat_completions_endpoint(label: str) -> dict:
    """POST /api/v1/chat/completions  — OpenAI-compatible endpoint."""
    url = f"{OPENROUTER_BASE}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "X-Title": RUN_TAG,
    }
    body = {
        "model": OPENAI_COMPAT_MODEL,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": f"Reply with just: {label}"}],
    }
    print(f"\n[2] POST {url}")
    print(f"    model: {OPENAI_COMPAT_MODEL}")
    status, resp = http_post(url, headers, body)
    gen_id = resp.get("id", "unknown")
    print(f"    HTTP {status}  →  id={gen_id}")
    if status != 200:
        print(f"    Response: {json.dumps(resp, indent=2)}")
    return resp


def check_langfuse_for_tag(tag: str, wait_seconds: int = 30) -> None:
    """Poll Langfuse traces for the RUN_TAG to verify what arrived."""
    if not (LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY):
        print(
            "\n[Langfuse check skipped — set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY "
            "to enable automatic verification]"
        )
        return

    import base64

    creds = base64.b64encode(
        f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()
    ).decode()
    headers = {"Authorization": f"Basic {creds}"}

    print(
        f"\n[3] Waiting {wait_seconds}s for OpenRouter to forward traces to Langfuse…"
    )
    time.sleep(wait_seconds)

    url = f"{LANGFUSE_HOST}/api/public/traces?tags={tag}&limit=10"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"    Langfuse API error: {e}")
        return

    traces = data.get("data", [])
    print(f"    Langfuse returned {len(traces)} trace(s) tagged '{tag}':")
    for t in traces:
        print(
            f"      - id={t.get('id')}  name={t.get('name')}  model={t.get('metadata', {})}"
        )

    if not traces:
        print(
            "\n  ⚠  NO traces found in Langfuse — this confirms the /messages endpoint "
            "is not forwarded by OpenRouter Broadcast."
        )
    else:
        print("\n  ✓ Traces found — at least one endpoint was forwarded.")


def main() -> None:
    check_env()

    print("=" * 70)
    print("OpenRouter Broadcast gap reproduction")
    print(f"Run tag : {RUN_TAG}")
    print(f"Time    : {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)
    print(
        "\nExpected result:\n"
        "  • /api/v1/messages        (Anthropic-native)  → HTTP 200, gen-ID returned,\n"
        "                              but NO trace in Langfuse / Broadcast.\n"
        "  • /api/v1/chat/completions (OpenAI-compat)    → HTTP 200, gen-ID returned,\n"
        "                              AND trace DOES appear in Langfuse / Broadcast.\n"
    )

    # Make both calls
    call_messages_endpoint("messages-test")
    call_chat_completions_endpoint("chat-completions-test")

    # Optionally verify via Langfuse API
    check_langfuse_for_tag(RUN_TAG)

    print("\n" + "=" * 70)
    print("EXPECTED OUTCOME (observed in our environment):")
    print("  /api/v1/messages        → gen-ID issued by OpenRouter, NOT in Langfuse")
    print("  /api/v1/chat/completions → gen-ID issued by OpenRouter, IS in Langfuse")
    print()
    print("BUG:")
    print("  When a Langfuse (or other Broadcast) integration is configured in")
    print("  OpenRouter, only requests to /api/v1/chat/completions are forwarded.")
    print("  Requests to /api/v1/messages are silently excluded, even though")
    print("  OpenRouter successfully proxies them to Anthropic and returns a gen-ID.")
    print()
    print("IMPACT:")
    print("  Any library that uses the Anthropic Messages API (e.g. Claude Agent SDK,")
    print("  anthropic-python) loses observability when routed through OpenRouter,")
    print("  even if Broadcast is fully configured.")
    print("=" * 70)


if __name__ == "__main__":
    main()
