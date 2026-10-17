"""Parse and triage the JSON findings produced by a Darkmoon scan."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField

SEVERITY_ORDER = ["info", "low", "medium", "high", "critical"]
PROVEN_STATUSES = {"exploited", "confirmed"}


class SeverityLevel(Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FailThreshold(Enum):
    NEVER = "never"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DarkmoonFinding(BaseModel):
    """One finding. Unknown fields are kept so nothing Darkmoon emits is lost."""

    model_config = ConfigDict(extra="allow")

    title: str = ""
    severity: str = ""
    cvss_score: float | None = None
    category: str | None = None
    status: str | None = None
    description: str | None = None
    endpoint: str | None = None
    cve: str | None = None


class DarkmoonFindingsEnvelope(BaseModel):
    """Findings wrapped in an object, as `findings` or as an API style `data` list."""

    findings: list[DarkmoonFinding] | None = None
    data: list[DarkmoonFinding] | None = None


_FINDING_LIST = TypeAdapter(list[DarkmoonFinding])


def parse_findings(raw: str) -> list[DarkmoonFinding]:
    """Accept a bare JSON array, or an object with a `findings` or `data` array."""
    try:
        return _FINDING_LIST.validate_json(raw)
    except ValidationError as list_error:
        try:
            envelope = DarkmoonFindingsEnvelope.model_validate_json(raw)
        except ValidationError:
            raise ValueError(
                "Not a Darkmoon findings document: expected a JSON array of "
                f"findings, or an object with a 'findings' or 'data' array. "
                f"{_first_error(list_error)}"
            ) from None
        found = envelope.findings if envelope.findings is not None else envelope.data
        if found is None:
            raise ValueError(
                "Not a Darkmoon findings document: the JSON object has neither a "
                "'findings' nor a 'data' array."
            ) from None
        return found


def _first_error(error: ValidationError) -> str:
    first = error.errors()[0]
    return f"({first['type']} at {'.'.join(str(p) for p in first['loc']) or 'root'})"


def severity_rank(finding: DarkmoonFinding) -> int:
    """Rank of the severity, -1 when missing or unrecognised."""
    value = finding.severity.strip().lower()
    return SEVERITY_ORDER.index(value) if value in SEVERITY_ORDER else -1


def count_by_severity(findings: list[DarkmoonFinding]) -> dict[str, int]:
    counts = {level: 0 for level in reversed(SEVERITY_ORDER)}
    for finding in findings:
        rank = severity_rank(finding)
        if rank >= 0:
            counts[SEVERITY_ORDER[rank]] += 1
    return counts


def render_markdown(findings: list[DarkmoonFinding], counts: dict[str, int]) -> str:
    summary = ", ".join(f"{n} {level}" for level, n in counts.items() if n)
    lines = [f"## Darkmoon findings: {len(findings)} ({summary or 'none rated'})"]
    if not findings:
        return lines[0]
    lines += [
        "",
        "| Severity | CVSS | Status | Title | Endpoint |",
        "|---|---|---|---|---|",
    ]
    for finding in findings:
        cvss = "" if finding.cvss_score is None else f"{finding.cvss_score:.1f}"
        row = [
            finding.severity.upper() or "UNRATED",
            cvss,
            finding.status or "",
            finding.title or "(untitled)",
            finding.endpoint or "",
        ]
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _cell(value: str) -> str:
    return " ".join(value.split()).replace("|", "\\|")


class DarkmoonFindingsParserBlock(Block):
    class Input(BlockSchemaInput):
        findings_json: str = SchemaField(
            description=(
                "The findings JSON from a Darkmoon scan: an array of findings, or an "
                "object holding them under 'findings' or 'data'."
            ),
            placeholder='[{"title": "SQL injection", "severity": "high", ...}]',
        )
        min_severity: SeverityLevel = SchemaField(
            description=(
                "Drop findings below this severity from the results. 'info' keeps "
                "everything, including findings without a recognised severity."
            ),
            default=SeverityLevel.INFO,
            advanced=False,
        )
        only_proven: bool = SchemaField(
            description=(
                "Keep only findings whose status is 'exploited' or 'confirmed', "
                "dropping 'unconfirmed' ones."
            ),
            default=False,
        )
        fail_on: FailThreshold = SchemaField(
            description=(
                "The gate output turns on when a kept finding is at or above this "
                "severity. Choose 'never' to disable the gate."
            ),
            default=FailThreshold.HIGH,
        )

    class Output(BlockSchemaOutput):
        findings: list[dict[str, Any]] = SchemaField(
            description="The kept findings, most severe first, with all their fields."
        )
        total: int = SchemaField(description="Number of findings kept.")
        severity_counts: dict[str, int] = SchemaField(
            description="Kept findings per severity (critical, high, medium, low, info)."
        )
        highest_severity: str = SchemaField(
            description="The most severe rating among kept findings, or 'none'."
        )
        gate_failed: bool = SchemaField(
            description="True when a kept finding reaches the fail_on severity."
        )
        markdown_report: str = SchemaField(
            description="A Markdown summary table, ready to post to chat or a ticket."
        )

    def __init__(self):
        super().__init__(
            id="61f9d8bc-c73c-48a2-a993-6f6d14aeafe8",
            description=(
                "Parses the JSON findings of a Darkmoon scan (open source autonomous "
                "AI pentest engine), filters them by severity and proof status, and "
                "returns counts, a pass/fail gate and a Markdown report. Findings "
                "may include false positives and need human review."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS, BlockCategory.DATA},
            input_schema=DarkmoonFindingsParserBlock.Input,
            output_schema=DarkmoonFindingsParserBlock.Output,
            test_input={
                "findings_json": (
                    '{"findings": ['
                    '{"title": "Reflected XSS", "severity": "medium", "cvss_score": 6.1,'
                    ' "status": "confirmed", "endpoint": "https://example.test/search"},'
                    '{"title": "SQL injection", "severity": "critical", "cvss_score": 9.8,'
                    ' "status": "exploited", "endpoint": "https://example.test/login"},'
                    '{"title": "Server banner", "severity": "info",'
                    ' "status": "unconfirmed"}]}'
                ),
                "min_severity": SeverityLevel.LOW.value,
                "only_proven": True,
                "fail_on": FailThreshold.HIGH.value,
            },
            test_output=[
                (
                    "findings",
                    [
                        {
                            "title": "SQL injection",
                            "severity": "critical",
                            "cvss_score": 9.8,
                            "category": None,
                            "status": "exploited",
                            "description": None,
                            "endpoint": "https://example.test/login",
                            "cve": None,
                        },
                        {
                            "title": "Reflected XSS",
                            "severity": "medium",
                            "cvss_score": 6.1,
                            "category": None,
                            "status": "confirmed",
                            "description": None,
                            "endpoint": "https://example.test/search",
                            "cve": None,
                        },
                    ],
                ),
                ("total", 2),
                (
                    "severity_counts",
                    {"critical": 1, "high": 0, "medium": 1, "low": 0, "info": 0},
                ),
                ("highest_severity", "critical"),
                ("gate_failed", True),
                (
                    "markdown_report",
                    "## Darkmoon findings: 2 (1 critical, 1 medium)\n\n"
                    "| Severity | CVSS | Status | Title | Endpoint |\n"
                    "|---|---|---|---|---|\n"
                    "| CRITICAL | 9.8 | exploited | SQL injection | "
                    "https://example.test/login |\n"
                    "| MEDIUM | 6.1 | confirmed | Reflected XSS | "
                    "https://example.test/search |",
                ),
            ],
            effect=BlockEffect.NONE,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        # "info" keeps everything, including findings with a missing or unknown
        # severity, so nothing is dropped silently unless a higher floor is chosen.
        floor = input_data.min_severity.value
        threshold = -1 if floor == "info" else SEVERITY_ORDER.index(floor)
        kept = [
            finding
            for finding in parse_findings(input_data.findings_json)
            if severity_rank(finding) >= threshold
            and (not input_data.only_proven or finding.status in PROVEN_STATUSES)
        ]
        kept.sort(key=severity_rank, reverse=True)
        counts = count_by_severity(kept)
        top = severity_rank(kept[0]) if kept else -1

        yield "findings", [finding.model_dump() for finding in kept]
        yield "total", len(kept)
        yield "severity_counts", counts
        yield "highest_severity", SEVERITY_ORDER[top] if top >= 0 else "none"
        yield "gate_failed", self._gate_failed(input_data.fail_on, top)
        yield "markdown_report", render_markdown(kept, counts)

    @staticmethod
    def _gate_failed(fail_on: FailThreshold, top_rank: int) -> bool:
        if fail_on is FailThreshold.NEVER:
            return False
        return top_rank >= SEVERITY_ORDER.index(fail_on.value)
