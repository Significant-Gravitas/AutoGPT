"""Tests for the enrichment blocks: whitespace-only identifier validation,
the any_flag fallback to enrich_profile, and error surfacing."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.dataforb2b._config import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_META_INPUT,
)
from backend.blocks.dataforb2b.enrich import (
    CompanyEnrichmentBlock,
    ProfileEnrichmentBlock,
)


async def _run_and_capture_payload(input_data):
    """Run the block with a mocked client, returning the payload sent to the API."""
    block = ProfileEnrichmentBlock()
    with patch.object(
        ProfileEnrichmentBlock,
        "fetch_profile",
        new=AsyncMock(return_value={"profile": {"name": "John Doe"}}),
    ) as mock_enrich:
        async for _ in block.run(input_data, credentials=TEST_CREDENTIALS):
            pass
        return mock_enrich.await_args.args[0]


@pytest.mark.asyncio
async def test_whitespace_only_identifier_raises_value_error():
    """Thread ED7l/DxBk: a whitespace-only profile_identifier must be rejected,
    not passed through as if it were a real value."""
    block = ProfileEnrichmentBlock()
    input_data = ProfileEnrichmentBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        profile_identifier="   ",
    )
    with pytest.raises(ValueError, match="profile_identifier"):
        async for _ in block.run(input_data, credentials=TEST_CREDENTIALS):
            pass


@pytest.mark.asyncio
async def test_identifier_is_stripped():
    input_data = ProfileEnrichmentBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        profile_identifier="  https://www.linkedin.com/in/johndoe  ",
        enrich_profile=True,
    )
    payload = await _run_and_capture_payload(input_data)
    assert payload["profile_identifier"] == "https://www.linkedin.com/in/johndoe"


@pytest.mark.asyncio
async def test_no_flags_set_falls_back_to_enrich_profile():
    """Thread ED6c: when none of the enrich_* flags are set, the block must still
    request the profile rather than sending an all-False payload."""
    input_data = ProfileEnrichmentBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        profile_identifier="https://www.linkedin.com/in/johndoe",
        enrich_profile=False,
        enrich_work_email=False,
        enrich_personal_email=False,
        enrich_github=False,
    )
    payload = await _run_and_capture_payload(input_data)
    assert payload["enrich_profile"] is True


@pytest.mark.asyncio
async def test_explicit_flag_is_respected_without_fallback():
    input_data = ProfileEnrichmentBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        profile_identifier="https://www.linkedin.com/in/johndoe",
        enrich_profile=False,
        enrich_work_email=True,
    )
    payload = await _run_and_capture_payload(input_data)
    assert payload["enrich_profile"] is False
    assert payload["enrich_work_email"] is True


async def _run_company_and_capture_identifier(input_data):
    """Run CompanyEnrichmentBlock with a mocked client, returning the identifier sent."""
    block = CompanyEnrichmentBlock()
    with patch.object(
        CompanyEnrichmentBlock,
        "fetch_company",
        new=AsyncMock(return_value={"name": "Google"}),
    ) as mock_enrich:
        async for _ in block.run(input_data, credentials=TEST_CREDENTIALS):
            pass
        return mock_enrich.await_args.args[0]


@pytest.mark.asyncio
async def test_company_whitespace_only_identifier_raises_value_error():
    """A whitespace-only company_identifier must be rejected, not sent as-is."""
    block = CompanyEnrichmentBlock()
    input_data = CompanyEnrichmentBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        company_identifier="   ",
    )
    with pytest.raises(ValueError, match="company_identifier"):
        async for _ in block.run(input_data, credentials=TEST_CREDENTIALS):
            pass


@pytest.mark.asyncio
async def test_company_identifier_is_stripped():
    input_data = CompanyEnrichmentBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        company_identifier="  google.com  ",
    )
    identifier = await _run_company_and_capture_identifier(input_data)
    assert identifier == "google.com"


@pytest.mark.asyncio
async def test_company_upstream_error_propagates_to_error_output():
    """The docs promise client/server errors surface via the framework 'error'
    output; that contract relies on run() letting the exception escape."""
    block = CompanyEnrichmentBlock()
    input_data = CompanyEnrichmentBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        company_identifier="google.com",
    )
    with patch.object(
        CompanyEnrichmentBlock,
        "fetch_company",
        new=AsyncMock(side_effect=RuntimeError("upstream 500")),
    ):
        with pytest.raises(RuntimeError, match="upstream 500"):
            async for _ in block.run(input_data, credentials=TEST_CREDENTIALS):
                pass
