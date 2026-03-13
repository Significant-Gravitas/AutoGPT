#!/usr/bin/env python3
"""
Run this script locally to apply the ABN Co-Navigator schema to Supabase.

Usage:
    SUPABASE_PAT=sbp_... python autogpt/coaching/setup_db.py

Or set SUPABASE_PAT in your .env file and run:
    python autogpt/coaching/setup_db.py
"""
import os
import sys
import urllib.request
import json
from pathlib import Path

# Load .env if present
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

PAT = os.environ.get("SUPABASE_PAT", "")
PROJECT_REF = "aakbytofflrctepuedyh"

if not PAT:
    print("ERROR: SUPABASE_PAT environment variable not set.")
    print("  Get your token at: https://supabase.com/dashboard/account/tokens")
    print("  Then run: SUPABASE_PAT=sbp_... python autogpt/coaching/setup_db.py")
    sys.exit(1)

SCHEMA_FILE = Path(__file__).parent / "supabase_schema.sql"
SQL = SCHEMA_FILE.read_text()

# Split on semicolons to run each statement individually
statements = [s.strip() for s in SQL.split(";") if s.strip() and not s.strip().startswith("--")]

API_URL = f"https://api.supabase.com/v1/projects/{PROJECT_REF}/database/query"

def run_sql(query: str) -> dict:
    payload = json.dumps({"query": query + ";"}).encode()
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {PAT}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return {"status": resp.status, "body": json.loads(resp.read().decode())}
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        return {"status": e.code, "error": body}
    except Exception as e:
        return {"status": 0, "error": str(e)}


print(f"Applying schema to Supabase project: {PROJECT_REF}")
print(f"Schema file: {SCHEMA_FILE}")
print("-" * 60)

errors = []
for i, stmt in enumerate(statements, 1):
    # Show first 60 chars of statement
    preview = stmt.replace("\n", " ")[:60]
    result = run_sql(stmt)
    status = result.get("status", 0)
    if status in (200, 201):
        print(f"  [{i:02d}] OK     {preview}")
    else:
        err = result.get("error", result.get("body", "unknown error"))
        # Ignore "already exists" errors — schema is idempotent
        if any(x in str(err) for x in ["already exists", "duplicate"]):
            print(f"  [{i:02d}] SKIP   {preview}  (already exists)")
        else:
            print(f"  [{i:02d}] ERROR  {preview}")
            print(f"         {err}")
            errors.append((i, stmt[:80], err))

print("-" * 60)
if errors:
    print(f"DONE with {len(errors)} error(s). Review above.")
    sys.exit(1)
else:
    print("Schema applied successfully. All tables ready.")
    print()
    print("Next step: add your SUPABASE_SERVICE_KEY to .env")
    print("  Found at: https://supabase.com/dashboard/project/aakbytofflrctepuedyh/settings/api")
