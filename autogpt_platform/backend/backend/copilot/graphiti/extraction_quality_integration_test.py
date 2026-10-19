"""Paid memory regression; opt in with RUN_GRAPHITI_LLM_EVAL=1 and API keys."""

import asyncio
import os
import re
from datetime import datetime

import pytest
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

from .client import _build_graphiti, _build_llm_config
from .ingest import CUSTOM_EXTRACTION_INSTRUCTIONS
from .llm_client import GraphitiOpenAIClient
from .memory_model import MemoryEnvelope
from .types import EDGE_TYPE_MAP, EDGE_TYPES, ENTITY_TYPES

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_GRAPHITI_LLM_EVAL") != "1",
        reason="Set RUN_GRAPHITI_LLM_EVAL=1 to make paid model and embedding calls",
    ),
]

_MESSAGES = [
    (
        "Prepare the supplier brief for Elm Studio. Our normal payment term is net 45 "
        "days, and our financial year ends in September. Please include those terms "
        "without claiming we have already signed anything."
    ),
    (
        "Use Lato for the documents. Dispatches leave our Omaha warehouse on Mondays "
        "and Wednesdays; the smallest order we accept is 18 cartons."
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source", [EpisodeType.message, EpisodeType.json, EpisodeType.text]
)
async def test_incidental_details_survive_as_durable_facts(clean_graph, source):
    driver, group_id = clean_graph
    llm_client = GraphitiOpenAIClient(config=_build_llm_config())
    client = _build_graphiti(group_id, llm_client, graph_driver=driver)
    try:
        await asyncio.wait_for(driver.ensure_indices(), timeout=30)
        await asyncio.wait_for(_ingest(client, group_id, source), timeout=180)
        rows, _, _ = await asyncio.wait_for(
            driver.execute_query("MATCH ()-[e:RELATES_TO]->() RETURN e.fact AS fact"),
            timeout=10,
        )
        facts = [row["fact"] for row in rows]
        issues = _extraction_issues(facts)
        assert (
            not issues
        ), f"Extraction issues: {issues}. Synthetic stored facts: {facts}"
    finally:
        await llm_client.client.close()
        await client.embedder.client.close()
        await client.cross_encoder.client.close()


async def _ingest(client: Graphiti, group_id: str, source: EpisodeType) -> None:
    for index, message in enumerate(_MESSAGES):
        content = (
            MemoryEnvelope(user="Jordan Rivera", content=message).model_dump_json()
            if source == EpisodeType.json
            else f"Jordan Rivera: {message}"
        )
        await client.add_episode(
            name=f"incidental_{index}",
            episode_body=content,
            source=source,
            source_description="Synthetic memory quality regression",
            reference_time=datetime.fromisoformat("2026-08-11T10:00:00+00:00"),
            group_id=group_id,
            entity_types=ENTITY_TYPES,
            edge_types=EDGE_TYPES,
            edge_type_map=EDGE_TYPE_MAP,
            custom_extraction_instructions=CUSTOM_EXTRACTION_INSTRUCTIONS,
        )


def _extraction_issues(facts: list[str]) -> list[str]:
    """Check concrete details and explicit unsupported claim forms in this scenario."""
    counterparty_details = (
        r"^\s*Elm Studio(?:['’]s\s+(?:normal\s+)?"
        r"(?:payment|financial|document|warehouse|dispatch|minimum)\b"
        r"|\s+(?:uses|dispatches|ships)\b)"
    )
    violations = {
        "unsupported signed-status claim": any(
            _has_signed_claim(fact) for fact in facts
        ),
        "misattributed company details": any(
            re.search(counterparty_details, fact, re.IGNORECASE) for fact in facts
        ),
    }
    return _missing_details(facts) + [
        name for name, found in violations.items() if found
    ]


def _missing_details(facts: list[str]) -> list[str]:
    dispatch = r"(?=.*\b(?:dispatch|ship|leav)\w*\b).*"
    payment_days = r"(?:\b45\b(?![.,]\d)|\bforty[- ]five\b)"
    expected = {
        "payment term": (
            rf"\bnet\s+{payment_days}|\bpayment[^.]{{0,90}}{payment_days}",
        ),
        "financial year end": (r"September",),
        "document font": (r"Lato",),
        "warehouse": (r"Omaha",),
        "dispatch days": (dispatch + r"\bMondays?\b", dispatch + r"\bWednesdays?\b"),
        "minimum order": (r"(?:18|eighteen)\s+cartons",),
    }
    return [
        name
        for name, patterns in expected.items()
        if not all(
            any(re.search(pattern, fact, re.IGNORECASE) for fact in facts)
            for pattern in patterns
        )
    ]


def _has_signed_claim(fact: str) -> bool:
    negated = (
        r"\b(?:not|never|\w+n['’]t)\s+(?:(?:yet|already|been|be|ever)\s+)*signed\b"
        r"|\bno\s+(?:agreement|contract|document)\s+"
        r"(?:(?:has|have|had|is|was|been|ever|yet)\s+)*signed\b"
        r"|\b(?:without|not(?:\s+to)?|never)\s+claim(?:ing)?\b[^.;!?]*?\bsigned\b"
        r"|\bsigned\s+no\s+(?:agreements?|contracts?|documents?)\b"
    )
    remaining = re.sub(negated, "", fact, flags=re.IGNORECASE)
    return re.search(r"\bsigned\b", remaining, re.IGNORECASE) is not None
