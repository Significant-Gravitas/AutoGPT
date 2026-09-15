"""Tier-0 retrieval benchmark: replay recorded ``find_block`` queries.

``t0_dataset.json`` holds 30 days of real ``find_block`` calls from prod and
dev.  Each is labelled with what the turn did next: the block it then ran
(``label_kind: block``), the platform tool it used instead (``tool``), the MCP
server it reached for (``mcp``), or nothing usable (``none``: it searched
again or gave up).  "Today" is the recorded ``find_block`` result list; the
registry is scored on the same query with the same context.
"""

import argparse
import json
from pathlib import Path

from pydantic import BaseModel, Field

from backend.copilot.capabilities.index import CapabilityIndex, SearchResult
from backend.copilot.capabilities.models import CapabilityEntry
from backend.copilot.capabilities.registry import get_registry
from backend.copilot.tools import TOOL_GROUPS, TOOL_REGISTRY

DATASET_PATH = Path(__file__).with_name("t0_dataset.json")
LABEL_KINDS = ("block", "tool", "mcp")
TOP_K = 5

# Query -> (context, acceptable top-3 answers as entry name or id).  Pinned
# regressions from the dossier: primitives, MCP servers, merged tools and
# graph-only blocks must resolve.
NAMED_CASES: dict[str, tuple[str, set[str]]] = {
    "execute code python": ("direct", {"bash_exec"}),
    "sentry": ("direct", {"mcp:mcp.sentry.dev"}),
    "linear issue": ("direct", {"LinearCreateIssueBlock", "mcp:mcp.linear.app"}),
    "http request": ("direct", {"SendWebRequestBlock"}),
    "send email": ("direct", {"GmailSendBlock", "SendEmailBlock"}),
    "google sheets add row": ("direct", {"GoogleSheetsAppendRowBlock"}),
    "OrchestratorBlock": ("graph", {"OrchestratorBlock"}),
}


class Case(BaseModel):
    query: str
    for_agent_generation: bool = False
    result_type: str | None = None
    returned: list[str] = Field(default_factory=list)
    label: str
    label_kind: str


class Stratum(BaseModel):
    n: int = 0
    hit1: int = 0
    hit5: int = 0
    miss: int = 0

    def add(self, rank: int | None, *, empty: bool) -> None:
        self.n += 1
        self.miss += empty
        if rank is not None:
            self.hit1 += rank == 0
            self.hit5 += rank < TOP_K

    def rate(self, field: str) -> float:
        return getattr(self, field) / self.n if self.n else 0.0


class Report(BaseModel):
    today: dict[str, Stratum]
    registry: dict[str, Stratum]
    named: dict[str, list[str]]
    misses: list[dict[str, object]]
    # Block cases find_block found (any rank) that the registry finds at 5.
    retained: int = 0
    retained_of: int = 0
    # Block cases find_block missed entirely that the registry finds at 5.
    gained: int = 0

    def retained_rate(self) -> float:
        return self.retained / self.retained_of if self.retained_of else 0.0

    def named_failures(self) -> dict[str, list[str]]:
        return {
            q: got for q, got in self.named.items() if not NAMED_CASES[q][1] & set(got)
        }


def load_cases(path: Path = DATASET_PATH) -> list[Case]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Case.model_validate(item) for item in data["items"]]


def evaluate(index: CapabilityIndex, cases: list[Case]) -> Report:
    today = {kind: Stratum() for kind in (*LABEL_KINDS, "all")}
    registry = {kind: Stratum() for kind in (*LABEL_KINDS, "all")}
    misses: list[dict[str, object]] = []
    retained = retained_of = gained = 0
    for case in cases:
        context = "graph" if case.for_agent_generation else "direct"
        result = index.search(case.query, context=context)
        today["all"].add(None, empty=case.result_type != "block_list")
        registry["all"].add(None, empty=not result.hits)
        if case.label_kind not in LABEL_KINDS:
            continue
        today_rank = _today_rank(case)
        rank = _registry_rank(result, case)
        today[case.label_kind].add(today_rank, empty=case.result_type != "block_list")
        registry[case.label_kind].add(rank, empty=not result.hits)
        if case.label_kind == "block":
            hit5 = rank is not None and rank < TOP_K
            retained_of += today_rank is not None
            retained += today_rank is not None and hit5
            gained += today_rank is None and hit5
        if rank is None or rank >= TOP_K:
            misses.append(
                {"query": case.query, "label": case.label, "got": result.names[:TOP_K]}
            )
    named = {q: _top3(index, q, context) for q, (context, _) in NAMED_CASES.items()}
    return Report(
        today=today,
        registry=registry,
        named=named,
        misses=misses,
        retained=retained,
        retained_of=retained_of,
        gained=gained,
    )


def _top3(index: CapabilityIndex, query: str, context: str) -> list[str]:
    result = index.search(query, context="graph" if context == "graph" else "direct")
    return result.names[:3] + result.ids[:3]


def format_report(report: Report) -> str:
    lines = [
        "| stratum | n | today r@1 | today r@5 | today none | registry r@1 | registry r@5 | registry none |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for kind in (*LABEL_KINDS, "all"):
        t, r = report.today[kind], report.registry[kind]
        lines.append(
            f"| {kind} | {r.n} | {_pct(t, 'hit1')} | {_pct(t, 'hit5')} | {_pct(t, 'miss')}"
            f" | {_pct(r, 'hit1')} | {_pct(r, 'hit5')} | {_pct(r, 'miss')} |"
        )
    lines.append(
        f"blocks find_block found that the registry keeps at 5: "
        f"{report.retained}/{report.retained_of} ({100 * report.retained_rate():.0f}%);"
        f" blocks it missed that the registry finds: {report.gained}"
    )
    failures = report.named_failures()
    lines.append(
        f"named cases: {len(NAMED_CASES) - len(failures)}/{len(NAMED_CASES)} pass"
    )
    lines += [f"  FAIL {q!r} -> {got}" for q, got in failures.items()]
    return "\n".join(lines)


def _pct(stratum: Stratum, field: str) -> str:
    return f"{100 * stratum.rate(field):.0f}%"


def _today_rank(case: Case) -> int | None:
    if case.label_kind != "block":
        return None  # find_block could never return a tool or an MCP server
    return case.returned.index(case.label) if case.label in case.returned else None


def _registry_rank(result: SearchResult, case: Case) -> int | None:
    for rank, hit in enumerate(result.hits):
        if _matches(hit.entry, case):
            return rank
    return None


def _matches(entry: CapabilityEntry, case: Case) -> bool:
    names = {entry.name} | {impl.name for impl in entry.implementations if impl.name}
    if case.label_kind == "mcp":
        return entry.id == f"mcp:{case.label}" or case.label in entry.tags
    if case.label_kind == "tool":
        return entry.kind == "tool" and entry.name == case.label
    return case.label in names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="also write the full report here")
    parser.add_argument("--misses", action="store_true", help="list every miss")
    args = parser.parse_args()
    report = evaluate(get_registry(TOOL_REGISTRY, TOOL_GROUPS), load_cases())
    print(format_report(report))
    if args.misses:
        for miss in report.misses:
            print(
                f"  {miss['query']!r:45} wanted {miss['label']!r:35} got {miss['got']}"
            )
    if args.json:
        args.json.write_text(report.model_dump_json(indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
