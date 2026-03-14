#!/usr/bin/env python3
"""
Apply the ABN Co-Navigator schema to Supabase.

Reads credentials from .env automatically. Run from the repo root:
    python autogpt/coaching/setup_db.py

Requires SUPABASE_PAT in .env (already set).
"""
import os
import sys
import urllib.request
import json
from pathlib import Path

# ── Load .env ────────────────────────────────────────────────────────────────
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

PAT         = os.environ.get("SUPABASE_PAT", "")
SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
PROJECT_REF = "aakbytofflrctepuedyh"

if not PAT:
    print("ERROR: SUPABASE_PAT not set in .env")
    print("  Get one at: https://supabase.com/dashboard/account/tokens")
    sys.exit(1)

# ── Load schema ───────────────────────────────────────────────────────────────
SCHEMA_FILE = Path(__file__).parent / "supabase_schema.sql"
SQL = SCHEMA_FILE.read_text()

# Strip comment-only lines, split on semicolons
statements = []
for raw in SQL.split(";"):
    stmt = "\n".join(
        line for line in raw.splitlines() if not line.strip().startswith("--")
    ).strip()
    if stmt:
        statements.append(stmt)

# ── Supabase Management API ───────────────────────────────────────────────────
MGMT_URL = f"https://api.supabase.com/v1/projects/{PROJECT_REF}/database/query"


def run_sql(query: str, token: str) -> dict:
    payload = json.dumps({"query": query + ";"}).encode()
    req = urllib.request.Request(
        MGMT_URL,
        data=payload,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return {"status": resp.status, "body": json.loads(resp.read().decode())}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "error": e.read().decode()}
    except Exception as e:
        return {"status": 0, "error": str(e)}


# ── Connectivity check ────────────────────────────────────────────────────────
print(f"Connecting to Supabase project: {PROJECT_REF}")
probe = run_sql("SELECT current_database()", PAT)
if probe.get("status") not in (200, 201):
    # Try service key as fallback
    if SERVICE_KEY:
        probe = run_sql("SELECT current_database()", SERVICE_KEY)
        if probe.get("status") in (200, 201):
            TOKEN = SERVICE_KEY
            print("  Auth: service role key")
        else:
            print(f"ERROR: Cannot connect. PAT response: {probe}")
            sys.exit(1)
    else:
        print(f"ERROR: Cannot connect. Response: {probe}")
        sys.exit(1)
else:
    TOKEN = PAT
    print("  Auth: personal access token")

print(f"  Schema: {SCHEMA_FILE.name}  ({len(statements)} statements)")
print("-" * 60)

# ── Execute schema ────────────────────────────────────────────────────────────
errors = []
for i, stmt in enumerate(statements, 1):
    preview = stmt.replace("\n", " ")[:65]
    result = run_sql(stmt, TOKEN)
    status = result.get("status", 0)
    if status in (200, 201):
        print(f"  [{i:02d}] OK    {preview}")
    else:
        err = str(result.get("error", result.get("body", "unknown")))
        if any(x in err for x in ["already exists", "duplicate"]):
            print(f"  [{i:02d}] SKIP  {preview}  (already exists)")
        else:
            print(f"  [{i:02d}] ERR   {preview}")
            print(f"        {err[:120]}")
            errors.append(i)

print("-" * 60)
if errors:
    print(f"Finished with {len(errors)} error(s) on statements: {errors}")
    sys.exit(1)
else:
    print("Schema applied. Tables: clients, coaching_sessions, key_results, obstacles")
    print("RLS enabled — only service_role key can read/write.")
    print()
    if not SERVICE_KEY:
        print("SUPABASE_SERVICE_KEY is not yet in .env.")
        print("  Get it: https://supabase.com/dashboard/project/aakbytofflrctepuedyh/settings/api")
        print("  Add to .env: SUPABASE_SERVICE_KEY=eyJ...")
    else:
        print("SUPABASE_SERVICE_KEY is set. Coaching API is ready to start.")
        print("  Run: python -m autogpt.coaching_server")
