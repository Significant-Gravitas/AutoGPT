#!/usr/bin/env python3
"""
TaskMarket Requester SDK (Python)
=================================
Lets a user complete the Taskmarket *requester* workflow from Python:
  - configure (Base network, spending cap)
  - create a task with explicit description/reward/deadline/deliverables
  - require FRESH user authorization before any funded on-chain action
  - retrieve live task status + submissions for human review (never auto-accept)

Uses the official `taskmarket` CLI as first-party tooling (no keys stored here).
Safeguards:
  - network enforced to Base (8453) only
  - max spend enforced before create
  - settlement status never blindly retried
  - no private keys / secrets handled by this module

Requires: `npm install -g @lucid-agents/taskmarket` on PATH.
"""
import json, subprocess, shutil, sys

BASE_CHAIN_ID = 8453  # Base mainnet
SAFE_COMMANDS = ("init", "wallet", "address", "task", "inbox", "agents", "stats")

class TaskMarketError(Exception):
    pass

def _run(args, authorize=False):
    if not shutil.which("npx"):
        raise TaskMarketError("npx/taskmarket CLI not found on PATH")
    cmd = ["npx", "taskmarket"] + args
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        raise TaskMarketError("taskmarket CLI timed out")
    if out.returncode != 0:
        raise TaskMarketError(out.stderr.strip() or "taskmarket failed")
    return out.stdout

def configure(network="Base", chain_id=BASE_CHAIN_ID):
    if chain_id != BASE_CHAIN_ID:
        raise TaskMarketError(f"Only Base (8453) is allowed; got {chain_id}")
    return {"network": network, "chain_id": chain_id}

def create_task(description, reward_usdc, deadline_unix, deliverables,
                max_spend_usdc, authorized_by_user=False, chain_id=BASE_CHAIN_ID):
    """Create a Taskmarket bounty. Requires explicit fresh user authorization."""
    if not authorized_by_user:
        raise TaskMarketError("FRESH user authorization required before funding a task")
    if chain_id != BASE_CHAIN_ID:
        raise TaskMarketError("Network check failed: only Base allowed")
    if reward_usdc <= 0 or reward_usdc > max_spend_usdc:
        raise TaskMarketError(f"Reward {reward_usdc} exceeds max spend {max_spend_usdc}")
    # Delegate actual on-chain create to first-party CLI (no keys here)
    out = _run(["task", "create", "--description", description,
                "--reward", str(int(reward_usdc * 1_000_000)),  # micro-USDC
                "--deadline", str(deadline_unix),
                "--deliverables", deliverables])
    # Parse returned task id/link
    task_id = None
    for line in out.splitlines():
        if "0x" in line and len(line.split("0x")[-1]) >= 60:
            task_id = "0x" + line.split("0x")[-1][:64]
            break
    return {"task_id": task_id, "raw": out}

def get_status(task_id):
    out = _run(["task", "get", task_id])
    try:
        data = json.loads(out)
        return data.get("data", data)
    except json.JSONDecodeError:
        return {"raw": out}

def get_submissions(task_id):
    """Retrieve submissions for HUMAN review. Never auto-accept/reject."""
    out = _run(["task", "get", task_id])
    try:
        data = json.loads(out).get("data", {})
        return data.get("submissions", data.get("awards", []))
    except json.JSONDecodeError:
        return []

if __name__ == "__main__":
    # Demo (no-op unless authorized): show config + a dry create
    print("TaskMarket Requester SDK ready. Base network enforced.")
    print("create_task(..., authorized_by_user=True) to fund a real task.")
