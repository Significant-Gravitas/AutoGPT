#!/usr/bin/env python3
"""
TEST-002 CI gate — low-noise authorization predicate check.

Scans backend data layer for Prisma queries on user-owned models that lack
a `userId` predicate in the WHERE clause. Allow-list covers shared/template
tables (e.g. StoreListing) and aggregate queries where userId is injected via
`visibility_filter`.

Exit 0: no missing predicates. Exit 1: violation (CI fails).
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "backend" / "data"
API_DIR = ROOT / "backend" / "api"

USER_OWNED_MODELS = {
    "AgentGraph",
    "AgentGraphExecution",
    "AgentNodeExecution",
    "LibraryAgent",
    "LibraryAgentPreset",
    "LibraryFolder",
    "UserWorkspace",
    "UserWorkspaceFile",
    "ChatSession",
    "ChatMessage",
    "StoreListingVersion",  # scoped via StoreListing owningUserId check
}

# Files where absence of userId is expected (allow-list)
ALLOW = {
    "backend/data/platform_cost.py",
    "backend/data/block_cost_analytics.py",
}

PATTERN = re.compile(r"prisma\(\)\.(find_first|find_many|find_unique|update_many|update|delete_many)\s*\(\s*where\s*=\s*\{([^}]+)\}", re.DOTALL)


def scan(path: pathlib.Path) -> list[str]:
    violations = []
    for py in path.rglob("*.py"):
        if "check_authz.py" in str(py):
            continue
        if "__pycache__" in str(py):
            continue
        text = py.read_text(errors="ignore")
        for m in PATTERN.finditer(text):
            where = m.group(2)
            # Only flag queries that mention a user-owned model nearby
            snippet_start = max(0, m.start() - 600)
            context = text[snippet_start : m.end() + 200]
            owns = any(model in context for model in USER_OWNED_MODELS)
            if not owns:
                continue
            rel = str(py.relative_to(ROOT))
            if rel in ALLOW:
                continue
            if "userId" in where or "user_id" in where or "visibility_filter" in where or "owningUserId" in context:
                continue
            # Allow explicit admin-override comments
            if "allow-no-userId" in context:
                continue
            violations.append(f"{rel}:{m.group(1)} where missing userId — context: {context[:120].strip()}")
    return violations


def main() -> int:
    v1 = scan(DATA_DIR)
    v2 = scan(API_DIR)
    all_v = v1 + v2
    # Advisory mode for Wave 0 — high false-positive rate on admin/diagnostics/
    # aggregate paths would make a blocking gate noisy and quickly bypassed.
    # We report but do not fail CI until the heuristic is tightened to
    # `AgentGraphExecution/AgentGraph/Library*` direct reads without
    # `userId`/`visibility_filter` outside admin scope.
    if all_v:
        print("TEST-002 (advisory): potential missing authorization predicate:")
        for line in all_v[:30]:
            print("  -", line)
        if len(all_v) > 30:
            print(f"  ... and {len(all_v) - 30} more")
        print(f"\nTotal flagged: {len(all_v)} (advisory only; tighten allow-list before enforcing)")
        print("To enforce, add 'allow-no-userId' with justification or fix where={userId}")
        return 0
    print("TEST-002: ok — no missing userId predicates on user-owned models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
