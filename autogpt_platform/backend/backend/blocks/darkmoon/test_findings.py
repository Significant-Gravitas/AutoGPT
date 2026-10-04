import json

import pytest

from backend.blocks.darkmoon.findings import (
    DarkmoonFindingsParserBlock,
    FailThreshold,
    SeverityLevel,
)

FINDINGS = [
    {
        "title": "Stored XSS | comment form",
        "severity": "high",
        "cvss_score": 7.2,
        "category": "xss_stored",
        "status": "confirmed",
        "endpoint": "https://app.test/comments",
        "discovered_by_agent": "nodejs",
    },
    {
        "title": "Open redirect",
        "severity": "low",
        "cvss_score": 3.1,
        "status": "unconfirmed",
    },
    {
        "title": "Outdated jQuery",
        "severity": "medium",
        "cvss_score": 5.0,
        "status": "unconfirmed",
        "cve": "CVE-2020-11022",
    },
]


async def run_block(findings_json: str, **overrides) -> dict:
    data = {"findings_json": findings_json, **overrides}
    block = DarkmoonFindingsParserBlock()
    return {
        name: value
        async for name, value in block.run(DarkmoonFindingsParserBlock.Input(**data))
    }


@pytest.mark.asyncio
async def test_bare_array_is_sorted_and_counted():
    out = await run_block(json.dumps(FINDINGS))
    assert [f["title"] for f in out["findings"]] == [
        "Stored XSS | comment form",
        "Outdated jQuery",
        "Open redirect",
    ]
    assert out["total"] == 3
    assert out["severity_counts"] == {
        "critical": 0,
        "high": 1,
        "medium": 1,
        "low": 1,
        "info": 0,
    }
    assert out["highest_severity"] == "high"


@pytest.mark.asyncio
async def test_extra_fields_are_preserved():
    out = await run_block(json.dumps(FINDINGS))
    assert out["findings"][0]["discovered_by_agent"] == "nodejs"
    assert out["findings"][0]["category"] == "xss_stored"


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["findings", "data"])
async def test_object_envelopes_are_accepted(key):
    out = await run_block(json.dumps({key: FINDINGS, "total": 3}))
    assert out["total"] == 3


@pytest.mark.asyncio
async def test_min_severity_filters_lower_findings():
    out = await run_block(json.dumps(FINDINGS), min_severity=SeverityLevel.MEDIUM.value)
    assert out["total"] == 2
    assert out["severity_counts"]["low"] == 0


@pytest.mark.asyncio
async def test_only_proven_drops_unconfirmed():
    out = await run_block(json.dumps(FINDINGS), only_proven=True)
    assert [f["status"] for f in out["findings"]] == ["confirmed"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fail_on, expected",
    [
        (FailThreshold.NEVER, False),
        (FailThreshold.LOW, True),
        (FailThreshold.HIGH, True),
        (FailThreshold.CRITICAL, False),
    ],
)
async def test_gate(fail_on, expected):
    out = await run_block(json.dumps(FINDINGS), fail_on=fail_on.value)
    assert out["gate_failed"] is expected


@pytest.mark.asyncio
async def test_gate_ignores_filtered_findings():
    out = await run_block(
        json.dumps(FINDINGS),
        min_severity=SeverityLevel.INFO.value,
        only_proven=True,
        fail_on=FailThreshold.CRITICAL.value,
    )
    assert out["gate_failed"] is False


@pytest.mark.asyncio
async def test_empty_list_passes():
    out = await run_block("[]")
    assert out["findings"] == []
    assert out["total"] == 0
    assert out["highest_severity"] == "none"
    assert out["gate_failed"] is False
    assert out["markdown_report"] == "## Darkmoon findings: 0 (none rated)"


@pytest.mark.asyncio
async def test_unrated_finding_is_kept_but_never_trips_the_gate():
    out = await run_block(
        json.dumps([{"title": "Odd one", "severity": "weird"}]),
        fail_on=FailThreshold.LOW.value,
    )
    assert out["total"] == 1
    assert out["gate_failed"] is False
    assert out["highest_severity"] == "none"
    assert "| UNRATED " not in out["markdown_report"]
    assert "| WEIRD |" in out["markdown_report"]


@pytest.mark.asyncio
async def test_unrated_finding_is_dropped_by_a_severity_filter():
    out = await run_block(
        json.dumps([{"title": "Odd one", "severity": "weird"}]),
        min_severity=SeverityLevel.LOW.value,
    )
    assert out["total"] == 0


@pytest.mark.asyncio
async def test_markdown_escapes_pipes_and_newlines():
    out = await run_block(json.dumps([FINDINGS[0] | {"endpoint": "a\nb"}]))
    assert "Stored XSS \\| comment form" in out["markdown_report"]
    assert "| a b |" in out["markdown_report"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    ["not json", '{"unrelated": 1}', '"just a string"', '[{"title": 5}]'],
)
async def test_invalid_documents_raise_a_clear_error(raw):
    with pytest.raises(ValueError, match="Darkmoon findings"):
        await run_block(raw)
