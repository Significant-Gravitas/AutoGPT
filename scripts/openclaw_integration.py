"""OpenClaw v2026.3.1 Integration for Auto-GPT.

Wraps the OpenClaw personal AI assistant gateway, providing:
  - Multi-channel message delivery (WhatsApp, Telegram, Slack, Discord, etc.)
  - Session management across channels
  - Agent-to-agent communication
  - Gateway health monitoring

OpenClaw runs as a local Node.js daemon (WebSocket on port 18789).
This module talks to it via HTTP REST + subprocess CLI fallback.

Ref: https://github.com/openclaw/openclaw/releases/tag/v2026.3.1
"""

import json
import subprocess
import os
import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENCLAW_VERSION = "2026.3.1"
OPENCLAW_GATEWAY_HOST = os.environ.get("OPENCLAW_HOST", "127.0.0.1")
OPENCLAW_GATEWAY_PORT = os.environ.get("OPENCLAW_PORT", "18789")
GATEWAY_BASE = f"http://{OPENCLAW_GATEWAY_HOST}:{OPENCLAW_GATEWAY_PORT}"

# Channels supported in v2026.3.1
SUPPORTED_CHANNELS = [
    "whatsapp", "telegram", "slack", "discord", "signal",
    "imessage", "google_chat", "line", "feishu", "email",
]


def _run_cli(*args, timeout=30):
    """Run an openclaw CLI command and return stdout.

    Falls back gracefully if openclaw is not installed.
    """
    try:
        result = subprocess.run(
            ["openclaw", *args],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            return f"Error (exit {result.returncode}): {result.stderr.strip()}"
        return result.stdout.strip()
    except FileNotFoundError:
        return ("Error: openclaw CLI not found. "
                "Install with: npm install -g openclaw@latest")
    except subprocess.TimeoutExpired:
        return "Error: openclaw command timed out."
    except Exception as e:
        return f"Error: {e}"


def _http_get(path):
    """GET request to the OpenClaw gateway."""
    try:
        import requests
        resp = requests.get(f"{GATEWAY_BASE}{path}", timeout=10)
        resp.raise_for_status()
        return resp.json() if resp.headers.get(
            "content-type", "").startswith("application/json") else resp.text
    except Exception as e:
        return {"error": str(e)}


def _http_post(path, data):
    """POST request to the OpenClaw gateway."""
    try:
        import requests
        resp = requests.post(
            f"{GATEWAY_BASE}{path}",
            json=data, timeout=30,
        )
        resp.raise_for_status()
        return resp.json() if resp.headers.get(
            "content-type", "").startswith("application/json") else resp.text
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Gateway health  (new in v2026.3.1: /health, /healthz, /ready, /readyz)
# ---------------------------------------------------------------------------

def openclaw_status():
    """Check OpenClaw gateway health and version.

    Returns:
        Status string with gateway health, version, and connectivity info.
    """
    health = _http_get("/healthz")

    if isinstance(health, dict) and "error" in health:
        # Gateway not reachable — try CLI
        cli_version = _run_cli("--version")
        return (
            f"OpenClaw gateway not reachable at {GATEWAY_BASE}.\n"
            f"CLI check: {cli_version}\n"
            f"Expected version: v{OPENCLAW_VERSION}\n"
            f"Start gateway with: openclaw gateway --port {OPENCLAW_GATEWAY_PORT}"
        )

    readiness = _http_get("/readyz")
    return (
        f"OpenClaw Gateway: HEALTHY\n"
        f"Endpoint: {GATEWAY_BASE}\n"
        f"Health: {json.dumps(health)}\n"
        f"Ready: {json.dumps(readiness)}\n"
        f"Target version: v{OPENCLAW_VERSION}"
    )


# ---------------------------------------------------------------------------
# Message delivery — multi-channel
# ---------------------------------------------------------------------------

def openclaw_send(channel, recipient, message):
    """Send a message through an OpenClaw channel.

    Args:
        channel: Channel name (e.g. 'telegram', 'slack', 'whatsapp').
        recipient: Channel-specific recipient ID or address.
        message: Text message to deliver.

    Returns:
        Delivery confirmation or error.
    """
    if channel not in SUPPORTED_CHANNELS:
        return (f"Error: unsupported channel '{channel}'. "
                f"Supported: {', '.join(SUPPORTED_CHANNELS)}")

    result = _run_cli(
        "message", "send",
        "--channel", channel,
        "--to", recipient,
        "--message", message,
    )
    return f"Sent via {channel} to {recipient}: {result}"


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def openclaw_sessions(action, session_id="", message=""):
    """Manage OpenClaw sessions.

    Args:
        action: One of 'list', 'history', 'send'.
        session_id: Required for 'history' and 'send'.
        message: Required for 'send'.

    Returns:
        Session data or confirmation.
    """
    if action == "list":
        return _run_cli("sessions", "list")

    if action == "history":
        if not session_id:
            return "Error: session_id required for history."
        return _run_cli("sessions", "history", "--id", session_id)

    if action == "send":
        if not session_id or not message:
            return "Error: session_id and message required for send."
        return _run_cli(
            "sessions", "send",
            "--id", session_id,
            "--message", message,
        )

    return f"Unknown session action '{action}'. Use: list, history, send."


# ---------------------------------------------------------------------------
# Agent interaction
# ---------------------------------------------------------------------------

def openclaw_agent(message, thinking="adaptive"):
    """Send a message to the OpenClaw agent runtime.

    Uses the v2026.3.1 adaptive thinking default for Claude 4.6 models.

    Args:
        message: Prompt or instruction for the agent.
        thinking: Thinking level — 'adaptive' (default), 'high', 'low', 'off'.

    Returns:
        Agent response.
    """
    return _run_cli(
        "agent",
        "--message", message,
        "--thinking", thinking,
    )


# ---------------------------------------------------------------------------
# Channel configuration helpers
# ---------------------------------------------------------------------------

def openclaw_channels():
    """List configured OpenClaw channels and their status.

    Returns:
        Channel configuration summary.
    """
    # Try gateway API first
    result = _http_get("/api/channels")
    if isinstance(result, dict) and "error" in result:
        # Fall back to CLI
        return _run_cli("channels", "list")
    return json.dumps(result, indent=2, ensure_ascii=False)
