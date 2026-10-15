"""Expert-hire retrieval benchmark: does ``find_capability`` surface the roster
expert a user is asking for, and stay quiet when they are not asking for one?

``expert_hire_dataset.json`` holds hand-written queries in four groups:
``plain`` names the role ("social media manager"), ``near`` describes the job
as a task ("someone to run my LinkedIn"), ``leap`` gives only the symptom ("I
need more followers"), and ``miss`` asks for something to run now, where no
expert belongs on top.  Labels are roster template names; any one of them
counts.  ``expert_roster.json`` is the roster snapshot the queries were
written against.  "Today" is the platform registry alone; "registry" layers
the roster on top the way a session does.  The model writes the actual
query, and when the user asked to hire it writes it the way Toran's session
did ("hire expert social media manager"), so each case is also scored with
that prefix.
"""

import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from backend.api.features.experts.models import (
    ExpertBundledSkill,
    ExpertTemplate,
    ExpertWorkflowRef,
)
from backend.copilot.capabilities.index import CapabilityIndex, SearchResult
from backend.copilot.capabilities.registry import build_entries
from backend.copilot.capabilities.sources import expert_entries
from backend.copilot.tools import TOOL_GROUPS, TOOL_REGISTRY

DATASET_PATH = Path(__file__).with_name("expert_hire_dataset.json")
ROSTER_PATH = Path(__file__).with_name("expert_roster.json")
Group = Literal["plain", "near", "leap", "miss"]
GROUPS: tuple[Group, ...] = ("plain", "near", "leap", "miss")
TOP_K = 5
HIRE_PREFIX = "hire expert "


class Case(BaseModel):
    query: str
    group: Group
    expect: list[str] = Field(default_factory=list)


class Stratum(BaseModel):
    n: int = 0
    hit1: int = 0
    hit3: int = 0
    hit5: int = 0
    # Miss cases only: an expert ranked first / anywhere in the top five.
    fp1: int = 0
    fp5: int = 0

    def rate(self, field: str) -> float:
        return getattr(self, field) / self.n if self.n else 0.0


class Failure(BaseModel):
    query: str
    group: Group
    expect: list[str]
    got: list[str]


class Report(BaseModel):
    today: dict[str, Stratum]
    registry: dict[str, Stratum]
    # ``find_capability(kind="expert")``: retrieval among experts alone.
    experts_only: dict[str, Stratum]
    # The query as the model writes it for a hire: ``HIRE_PREFIX`` + query.
    hire_phrased: dict[str, Stratum]
    failures: list[Failure]
    # Templates no plain-role query puts in the top five.
    unreachable: list[str]


def load_cases(path: Path = DATASET_PATH) -> list[Case]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Case.model_validate(item) for item in data["items"]]


def load_roster(path: Path = ROSTER_PATH) -> list[ExpertTemplate]:
    """The snapshot as the templates ``with_bundled_skills`` returns, with the
    fields the entries never read left empty."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [_template(item) for item in data["templates"]]


def platform_index() -> CapabilityIndex:
    # Same catalogue as the registry benchmark, so a machine without provider
    # secrets competes the roster against every block CI sees.
    return CapabilityIndex(
        build_entries(TOOL_REGISTRY, TOOL_GROUPS, include_disabled_blocks=True)
    )


def evaluate(
    today: CapabilityIndex, templates: list[ExpertTemplate], cases: list[Case]
) -> Report:
    registry = today.with_entries(expert_entries(templates, hired=[]))
    report = Report(
        today={g: Stratum() for g in GROUPS},
        registry={g: Stratum() for g in GROUPS},
        experts_only={g: Stratum() for g in GROUPS},
        hire_phrased={g: Stratum() for g in GROUPS},
        failures=[],
        unreachable=[],
    )
    reached: set[str] = set()
    for case in cases:
        result = registry.search(case.query, context="direct")
        _score(report.today[case.group], today.search(case.query), case)
        _score(report.registry[case.group], result, case)
        _score(
            report.experts_only[case.group],
            registry.search(case.query, context="direct", kind="expert"),
            case,
        )
        _score(
            report.hire_phrased[case.group],
            registry.search(HIRE_PREFIX + case.query, context="direct"),
            case,
        )
        if case.group == "plain":
            reached |= set(_expert_names(result)[:TOP_K]) & set(case.expect)
        if _failed(result, case):
            report.failures.append(
                Failure(
                    query=case.query,
                    group=case.group,
                    expect=case.expect,
                    got=[_label(hit.entry) for hit in result.hits[:TOP_K]],
                )
            )
    report.unreachable = sorted({t.name for t in templates} - reached)
    return report


def format_report(report: Report) -> str:
    lines = [
        "| group | n | today h@5 | registry h@1 | registry h@3 | registry h@5 "
        "| kind=expert h@1 | kind=expert h@5 | hire-phrased h@1 | hire-phrased h@5 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for group in GROUPS[:-1]:
        t, r, e, h = (
            report.today[group],
            report.registry[group],
            report.experts_only[group],
            report.hire_phrased[group],
        )
        lines.append(
            f"| {group} | {r.n} | {_pct(t, 'hit5')} | {_pct(r, 'hit1')} "
            f"| {_pct(r, 'hit3')} | {_pct(r, 'hit5')} | {_pct(e, 'hit1')} "
            f"| {_pct(e, 'hit5')} | {_pct(h, 'hit1')} | {_pct(h, 'hit5')} |"
        )
    t, r = report.today["miss"], report.registry["miss"]
    lines.append(
        f"miss (n={r.n}): an expert ranked first {_pct(r, 'fp1')}, in the top "
        f"{TOP_K} {_pct(r, 'fp5')} (today {_pct(t, 'fp5')})"
    )
    lines.append(
        "templates no plain query reaches at 5: "
        + (", ".join(report.unreachable) or "none")
    )
    lines += [
        f"  FAIL [{f.group}] {f.query!r} wanted {f.expect or 'no expert'} got {f.got}"
        for f in report.failures
    ]
    return "\n".join(lines)


def _score(stratum: Stratum, result: SearchResult, case: Case) -> None:
    stratum.n += 1
    if case.group == "miss":
        stratum.fp1 += bool(result.hits) and result.hits[0].entry.kind == "expert"
        stratum.fp5 += any(h.entry.kind == "expert" for h in result.hits[:TOP_K])
        return
    rank = _rank(result, case)
    if rank is not None:
        stratum.hit1 += rank < 1
        stratum.hit3 += rank < 3
        stratum.hit5 += rank < TOP_K


def _rank(result: SearchResult, case: Case) -> int | None:
    for rank, hit in enumerate(result.hits):
        if hit.entry.kind == "expert" and hit.entry.name in case.expect:
            return rank
    return None


def _failed(result: SearchResult, case: Case) -> bool:
    if case.group == "miss":
        return any(h.entry.kind == "expert" for h in result.hits[:TOP_K])
    rank = _rank(result, case)
    return rank is None or rank >= TOP_K


def _expert_names(result: SearchResult) -> list[str]:
    return [h.entry.name if h.entry.kind == "expert" else "" for h in result.hits]


def _label(entry) -> str:
    return f"expert:{entry.name}" if entry.kind == "expert" else entry.name


def _pct(stratum: Stratum, field: str) -> str:
    return f"{100 * stratum.rate(field):.0f}%"


def _template(item: dict) -> ExpertTemplate:
    return ExpertTemplate(
        id=item["id"],
        name=item["name"],
        avatar_url=None,
        role=item["role"],
        job_title=item.get("job_title"),
        tagline=item.get("tagline"),
        bio=None,
        skills=[],
        categories=item.get("categories") or [],
        identity="",
        voice_preferences="",
        boundaries="",
        protected_soul_rules=[],
        is_template=True,
        source_template_id=None,
        is_archived=False,
        workflows=[
            ExpertWorkflowRef(
                id=f"{item['id']}-wf-{i}",
                store_listing_version_id=None,
                library_agent_id=None,
                graph_id=None,
                name=name,
                description=None,
            )
            for i, name in enumerate(item.get("workflows") or [])
        ],
        bundled_skills=[
            ExpertBundledSkill(
                id=f"{item['id']}-skill-{i}", slug="", title=title, description=""
            )
            for i, title in enumerate(item.get("skills") or [])
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="also write the full report here")
    args = parser.parse_args()
    report = evaluate(platform_index(), load_roster(), load_cases())
    print(format_report(report))
    if args.json:
        args.json.write_text(report.model_dump_json(indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
