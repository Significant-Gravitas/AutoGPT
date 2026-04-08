"""Unit tests for merge_stats cost tracking in individual blocks.

Covers the exa code_context, exa contents, and apollo organization blocks
to verify provider cost is correctly extracted and reported.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, NodeExecutionStats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEST_EXA_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="exa",
    api_key=SecretStr("mock-exa-api-key"),
    title="Mock Exa API key",
    expires_at=None,
)

TEST_EXA_CREDENTIALS_INPUT = {
    "provider": TEST_EXA_CREDENTIALS.provider,
    "id": TEST_EXA_CREDENTIALS.id,
    "type": TEST_EXA_CREDENTIALS.type,
    "title": TEST_EXA_CREDENTIALS.title,
}


# ---------------------------------------------------------------------------
# ExaCodeContextBlock — cost_dollars is a string like "0.005"
# ---------------------------------------------------------------------------


class TestExaCodeContextBlockCostTracking:
    @pytest.mark.asyncio
    async def test_merge_stats_called_with_float_cost(self):
        """float(cost_dollars) parsed from API string and passed to merge_stats."""
        from backend.blocks.exa.code_context import ExaCodeContextBlock

        block = ExaCodeContextBlock()

        api_response = {
            "requestId": "req-1",
            "query": "how to use hooks",
            "response": "Here are some examples...",
            "resultsCount": 3,
            "costDollars": "0.005",
            "searchTime": 1.2,
            "outputTokens": 100,
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = api_response

        accumulated: list[NodeExecutionStats] = []

        with (
            patch(
                "backend.blocks.exa.code_context.Requests.post",
                new_callable=AsyncMock,
                return_value=mock_resp,
            ),
            patch.object(
                block, "merge_stats", side_effect=lambda s: accumulated.append(s)
            ),
        ):
            input_data = ExaCodeContextBlock.Input(
                query="how to use hooks",
                credentials=TEST_EXA_CREDENTIALS_INPUT,  # type: ignore[arg-type]
            )
            results = []
            async for output in block.run(
                input_data,
                credentials=TEST_EXA_CREDENTIALS,
            ):
                results.append(output)

        assert len(accumulated) == 1
        assert accumulated[0].provider_cost == pytest.approx(0.005)

    @pytest.mark.asyncio
    async def test_invalid_cost_dollars_does_not_raise(self):
        """When cost_dollars cannot be parsed as float, merge_stats is not called."""
        from backend.blocks.exa.code_context import ExaCodeContextBlock

        block = ExaCodeContextBlock()

        api_response = {
            "requestId": "req-2",
            "query": "query",
            "response": "response",
            "resultsCount": 0,
            "costDollars": "N/A",
            "searchTime": 0.5,
            "outputTokens": 0,
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = api_response

        merge_calls: list[NodeExecutionStats] = []

        with (
            patch(
                "backend.blocks.exa.code_context.Requests.post",
                new_callable=AsyncMock,
                return_value=mock_resp,
            ),
            patch.object(
                block, "merge_stats", side_effect=lambda s: merge_calls.append(s)
            ),
        ):
            input_data = ExaCodeContextBlock.Input(
                query="query",
                credentials=TEST_EXA_CREDENTIALS_INPUT,  # type: ignore[arg-type]
            )
            async for _ in block.run(
                input_data,
                credentials=TEST_EXA_CREDENTIALS,
            ):
                pass

        assert merge_calls == []

    @pytest.mark.asyncio
    async def test_zero_cost_is_tracked(self):
        """A zero cost_dollars string '0.0' should still be recorded."""
        from backend.blocks.exa.code_context import ExaCodeContextBlock

        block = ExaCodeContextBlock()

        api_response = {
            "requestId": "req-3",
            "query": "query",
            "response": "...",
            "resultsCount": 1,
            "costDollars": "0.0",
            "searchTime": 0.1,
            "outputTokens": 10,
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = api_response

        accumulated: list[NodeExecutionStats] = []

        with (
            patch(
                "backend.blocks.exa.code_context.Requests.post",
                new_callable=AsyncMock,
                return_value=mock_resp,
            ),
            patch.object(
                block, "merge_stats", side_effect=lambda s: accumulated.append(s)
            ),
        ):
            input_data = ExaCodeContextBlock.Input(
                query="query",
                credentials=TEST_EXA_CREDENTIALS_INPUT,  # type: ignore[arg-type]
            )
            async for _ in block.run(
                input_data,
                credentials=TEST_EXA_CREDENTIALS,
            ):
                pass

        assert len(accumulated) == 1
        assert accumulated[0].provider_cost == 0.0


# ---------------------------------------------------------------------------
# ExaContentsBlock — response.cost_dollars.total (CostDollars model)
# ---------------------------------------------------------------------------


class TestExaContentsBlockCostTracking:
    @pytest.mark.asyncio
    async def test_merge_stats_called_with_cost_dollars_total(self):
        """provider_cost equals response.cost_dollars.total when present."""
        from backend.blocks.exa.contents import ExaContentsBlock
        from backend.blocks.exa.helpers import CostDollars

        block = ExaContentsBlock()

        cost_dollars = CostDollars(total=0.012)

        mock_response = MagicMock()
        mock_response.results = []
        mock_response.context = None
        mock_response.statuses = None
        mock_response.cost_dollars = cost_dollars

        accumulated: list[NodeExecutionStats] = []

        with (
            patch(
                "backend.blocks.exa.contents.AsyncExa",
                return_value=MagicMock(
                    get_contents=AsyncMock(return_value=mock_response)
                ),
            ),
            patch.object(
                block, "merge_stats", side_effect=lambda s: accumulated.append(s)
            ),
        ):
            input_data = ExaContentsBlock.Input(
                urls=["https://example.com"],
                credentials=TEST_EXA_CREDENTIALS_INPUT,  # type: ignore[arg-type]
            )
            async for _ in block.run(
                input_data,
                credentials=TEST_EXA_CREDENTIALS,
            ):
                pass

        assert len(accumulated) == 1
        assert accumulated[0].provider_cost == pytest.approx(0.012)

    @pytest.mark.asyncio
    async def test_no_merge_stats_when_cost_dollars_absent(self):
        """When response.cost_dollars is None, merge_stats is not called."""
        from backend.blocks.exa.contents import ExaContentsBlock

        block = ExaContentsBlock()

        mock_response = MagicMock()
        mock_response.results = []
        mock_response.context = None
        mock_response.statuses = None
        mock_response.cost_dollars = None

        accumulated: list[NodeExecutionStats] = []

        with (
            patch(
                "backend.blocks.exa.contents.AsyncExa",
                return_value=MagicMock(
                    get_contents=AsyncMock(return_value=mock_response)
                ),
            ),
            patch.object(
                block, "merge_stats", side_effect=lambda s: accumulated.append(s)
            ),
        ):
            input_data = ExaContentsBlock.Input(
                urls=["https://example.com"],
                credentials=TEST_EXA_CREDENTIALS_INPUT,  # type: ignore[arg-type]
            )
            async for _ in block.run(
                input_data,
                credentials=TEST_EXA_CREDENTIALS,
            ):
                pass

        assert accumulated == []


# ---------------------------------------------------------------------------
# SearchOrganizationsBlock — provider_cost = float(len(organizations))
# ---------------------------------------------------------------------------


class TestSearchOrganizationsBlockCostTracking:
    @pytest.mark.asyncio
    async def test_merge_stats_called_with_org_count(self):
        """provider_cost == number of returned organizations, type == 'items'."""
        from backend.blocks.apollo._auth import TEST_CREDENTIALS as APOLLO_CREDS
        from backend.blocks.apollo._auth import (
            TEST_CREDENTIALS_INPUT as APOLLO_CREDS_INPUT,
        )
        from backend.blocks.apollo.models import Organization
        from backend.blocks.apollo.organization import SearchOrganizationsBlock

        block = SearchOrganizationsBlock()

        fake_orgs = [Organization(id=str(i), name=f"Org{i}") for i in range(3)]

        accumulated: list[NodeExecutionStats] = []

        with (
            patch.object(
                SearchOrganizationsBlock,
                "search_organizations",
                new_callable=AsyncMock,
                return_value=fake_orgs,
            ),
            patch.object(
                block, "merge_stats", side_effect=lambda s: accumulated.append(s)
            ),
        ):
            input_data = SearchOrganizationsBlock.Input(
                credentials=APOLLO_CREDS_INPUT,  # type: ignore[arg-type]
            )
            results = []
            async for output in block.run(
                input_data,
                credentials=APOLLO_CREDS,
            ):
                results.append(output)

        assert len(accumulated) == 1
        assert accumulated[0].provider_cost == pytest.approx(3.0)
        assert accumulated[0].provider_cost_type == "items"

    @pytest.mark.asyncio
    async def test_empty_org_list_tracks_zero(self):
        """An empty organization list results in provider_cost=0.0."""
        from backend.blocks.apollo._auth import TEST_CREDENTIALS as APOLLO_CREDS
        from backend.blocks.apollo._auth import (
            TEST_CREDENTIALS_INPUT as APOLLO_CREDS_INPUT,
        )
        from backend.blocks.apollo.organization import SearchOrganizationsBlock

        block = SearchOrganizationsBlock()
        accumulated: list[NodeExecutionStats] = []

        with (
            patch.object(
                SearchOrganizationsBlock,
                "search_organizations",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch.object(
                block, "merge_stats", side_effect=lambda s: accumulated.append(s)
            ),
        ):
            input_data = SearchOrganizationsBlock.Input(
                credentials=APOLLO_CREDS_INPUT,  # type: ignore[arg-type]
            )
            async for _ in block.run(
                input_data,
                credentials=APOLLO_CREDS,
            ):
                pass

        assert len(accumulated) == 1
        assert accumulated[0].provider_cost == 0.0
        assert accumulated[0].provider_cost_type == "items"


# ---------------------------------------------------------------------------
# JinaEmbeddingBlock — token count from usage.total_tokens
# ---------------------------------------------------------------------------


class TestJinaEmbeddingBlockCostTracking:
    @pytest.mark.asyncio
    async def test_merge_stats_called_with_token_count(self):
        """provider token count is recorded when API returns usage.total_tokens."""
        from backend.blocks.jina._auth import TEST_CREDENTIALS as JINA_CREDS
        from backend.blocks.jina._auth import TEST_CREDENTIALS_INPUT as JINA_CREDS_INPUT
        from backend.blocks.jina.embeddings import JinaEmbeddingBlock

        block = JinaEmbeddingBlock()

        api_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
            "usage": {"total_tokens": 42},
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = api_response

        accumulated: list[NodeExecutionStats] = []

        with (
            patch(
                "backend.blocks.jina.embeddings.Requests.post",
                new_callable=AsyncMock,
                return_value=mock_resp,
            ),
            patch.object(
                block, "merge_stats", side_effect=lambda s: accumulated.append(s)
            ),
        ):
            input_data = JinaEmbeddingBlock.Input(
                texts=["hello world"],
                credentials=JINA_CREDS_INPUT,  # type: ignore[arg-type]
            )
            async for _ in block.run(input_data, credentials=JINA_CREDS):
                pass

        assert len(accumulated) == 1
        assert accumulated[0].input_token_count == 42

    @pytest.mark.asyncio
    async def test_no_merge_stats_when_usage_absent(self):
        """When API response omits usage field, merge_stats is not called."""
        from backend.blocks.jina._auth import TEST_CREDENTIALS as JINA_CREDS
        from backend.blocks.jina._auth import TEST_CREDENTIALS_INPUT as JINA_CREDS_INPUT
        from backend.blocks.jina.embeddings import JinaEmbeddingBlock

        block = JinaEmbeddingBlock()

        api_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = api_response

        accumulated: list[NodeExecutionStats] = []

        with (
            patch(
                "backend.blocks.jina.embeddings.Requests.post",
                new_callable=AsyncMock,
                return_value=mock_resp,
            ),
            patch.object(
                block, "merge_stats", side_effect=lambda s: accumulated.append(s)
            ),
        ):
            input_data = JinaEmbeddingBlock.Input(
                texts=["hello"],
                credentials=JINA_CREDS_INPUT,  # type: ignore[arg-type]
            )
            async for _ in block.run(input_data, credentials=JINA_CREDS):
                pass

        assert accumulated == []


# ---------------------------------------------------------------------------
# UnrealTextToSpeechBlock — character count from input text length
# ---------------------------------------------------------------------------


class TestUnrealTextToSpeechBlockCostTracking:
    @pytest.mark.asyncio
    async def test_merge_stats_called_with_character_count(self):
        """provider_cost equals len(text) with type='characters'."""
        from backend.blocks.text_to_speech_block import TEST_CREDENTIALS as TTS_CREDS
        from backend.blocks.text_to_speech_block import (
            TEST_CREDENTIALS_INPUT as TTS_CREDS_INPUT,
        )
        from backend.blocks.text_to_speech_block import UnrealTextToSpeechBlock

        block = UnrealTextToSpeechBlock()
        test_text = "Hello, world!"

        with (
            patch.object(
                UnrealTextToSpeechBlock,
                "call_unreal_speech_api",
                new_callable=AsyncMock,
                return_value={"OutputUri": "https://example.com/audio.mp3"},
            ),
            patch.object(block, "merge_stats") as mock_merge,
        ):
            input_data = UnrealTextToSpeechBlock.Input(
                text=test_text,
                credentials=TTS_CREDS_INPUT,  # type: ignore[arg-type]
            )
            async for _ in block.run(input_data, credentials=TTS_CREDS):
                pass

        mock_merge.assert_called_once()
        stats = mock_merge.call_args[0][0]
        assert stats.provider_cost == float(len(test_text))
        assert stats.provider_cost_type == "characters"

    @pytest.mark.asyncio
    async def test_empty_text_gives_zero_characters(self):
        """An empty text string results in provider_cost=0.0."""
        from backend.blocks.text_to_speech_block import TEST_CREDENTIALS as TTS_CREDS
        from backend.blocks.text_to_speech_block import (
            TEST_CREDENTIALS_INPUT as TTS_CREDS_INPUT,
        )
        from backend.blocks.text_to_speech_block import UnrealTextToSpeechBlock

        block = UnrealTextToSpeechBlock()

        with (
            patch.object(
                UnrealTextToSpeechBlock,
                "call_unreal_speech_api",
                new_callable=AsyncMock,
                return_value={"OutputUri": "https://example.com/audio.mp3"},
            ),
            patch.object(block, "merge_stats") as mock_merge,
        ):
            input_data = UnrealTextToSpeechBlock.Input(
                text="",
                credentials=TTS_CREDS_INPUT,  # type: ignore[arg-type]
            )
            async for _ in block.run(input_data, credentials=TTS_CREDS):
                pass

        mock_merge.assert_called_once()
        stats = mock_merge.call_args[0][0]
        assert stats.provider_cost == 0.0
        assert stats.provider_cost_type == "characters"


# ---------------------------------------------------------------------------
# GoogleMapsSearchBlock — item count from search_places results
# ---------------------------------------------------------------------------


class TestGoogleMapsSearchBlockCostTracking:
    @pytest.mark.asyncio
    async def test_merge_stats_called_with_place_count(self):
        """provider_cost equals number of returned places, type == 'items'."""
        from backend.blocks.google_maps import TEST_CREDENTIALS as MAPS_CREDS
        from backend.blocks.google_maps import (
            TEST_CREDENTIALS_INPUT as MAPS_CREDS_INPUT,
        )
        from backend.blocks.google_maps import GoogleMapsSearchBlock

        block = GoogleMapsSearchBlock()

        fake_places = [{"name": f"Place{i}", "address": f"Addr{i}"} for i in range(4)]
        accumulated: list[NodeExecutionStats] = []

        with (
            patch.object(
                GoogleMapsSearchBlock,
                "search_places",
                return_value=fake_places,
            ),
            patch.object(
                block, "merge_stats", side_effect=lambda s: accumulated.append(s)
            ),
        ):
            input_data = GoogleMapsSearchBlock.Input(
                query="coffee shops",
                credentials=MAPS_CREDS_INPUT,  # type: ignore[arg-type]
            )
            async for _ in block.run(input_data, credentials=MAPS_CREDS):
                pass

        assert len(accumulated) == 1
        assert accumulated[0].provider_cost == 4.0
        assert accumulated[0].provider_cost_type == "items"

    @pytest.mark.asyncio
    async def test_empty_results_tracks_zero(self):
        """Zero places returned results in provider_cost=0.0."""
        from backend.blocks.google_maps import TEST_CREDENTIALS as MAPS_CREDS
        from backend.blocks.google_maps import (
            TEST_CREDENTIALS_INPUT as MAPS_CREDS_INPUT,
        )
        from backend.blocks.google_maps import GoogleMapsSearchBlock

        block = GoogleMapsSearchBlock()
        accumulated: list[NodeExecutionStats] = []

        with (
            patch.object(
                GoogleMapsSearchBlock,
                "search_places",
                return_value=[],
            ),
            patch.object(
                block, "merge_stats", side_effect=lambda s: accumulated.append(s)
            ),
        ):
            input_data = GoogleMapsSearchBlock.Input(
                query="nothing here",
                credentials=MAPS_CREDS_INPUT,  # type: ignore[arg-type]
            )
            async for _ in block.run(input_data, credentials=MAPS_CREDS):
                pass

        assert len(accumulated) == 1
        assert accumulated[0].provider_cost == 0.0
        assert accumulated[0].provider_cost_type == "items"


# ---------------------------------------------------------------------------
# SmartLeadAddLeadsBlock — item count from lead_list length
# ---------------------------------------------------------------------------


class TestSmartLeadAddLeadsBlockCostTracking:
    @pytest.mark.asyncio
    async def test_merge_stats_called_with_lead_count(self):
        """provider_cost equals number of leads uploaded, type == 'items'."""
        from backend.blocks.smartlead._auth import TEST_CREDENTIALS as SL_CREDS
        from backend.blocks.smartlead._auth import (
            TEST_CREDENTIALS_INPUT as SL_CREDS_INPUT,
        )
        from backend.blocks.smartlead.campaign import AddLeadToCampaignBlock
        from backend.blocks.smartlead.models import (
            AddLeadsToCampaignResponse,
            LeadInput,
        )

        block = AddLeadToCampaignBlock()

        fake_leads = [
            LeadInput(first_name="Alice", last_name="A", email="alice@example.com"),
            LeadInput(first_name="Bob", last_name="B", email="bob@example.com"),
        ]
        fake_response = AddLeadsToCampaignResponse(
            ok=True,
            upload_count=2,
            total_leads=2,
            block_count=0,
            duplicate_count=0,
            invalid_email_count=0,
            invalid_emails=[],
            already_added_to_campaign=0,
            unsubscribed_leads=[],
            is_lead_limit_exhausted=False,
            lead_import_stopped_count=0,
            bounce_count=0,
        )
        accumulated: list[NodeExecutionStats] = []

        with (
            patch.object(
                AddLeadToCampaignBlock,
                "add_leads_to_campaign",
                new_callable=AsyncMock,
                return_value=fake_response,
            ),
            patch.object(
                block, "merge_stats", side_effect=lambda s: accumulated.append(s)
            ),
        ):
            input_data = AddLeadToCampaignBlock.Input(
                campaign_id=123,
                lead_list=fake_leads,
                credentials=SL_CREDS_INPUT,  # type: ignore[arg-type]
            )
            async for _ in block.run(input_data, credentials=SL_CREDS):
                pass

        assert len(accumulated) == 1
        assert accumulated[0].provider_cost == 2.0
        assert accumulated[0].provider_cost_type == "items"


# ---------------------------------------------------------------------------
# SearchPeopleBlock — item count from people list length
# ---------------------------------------------------------------------------


class TestSearchPeopleBlockCostTracking:
    @pytest.mark.asyncio
    async def test_merge_stats_called_with_people_count(self):
        """provider_cost equals number of returned people, type == 'items'."""
        from backend.blocks.apollo._auth import TEST_CREDENTIALS as APOLLO_CREDS
        from backend.blocks.apollo._auth import (
            TEST_CREDENTIALS_INPUT as APOLLO_CREDS_INPUT,
        )
        from backend.blocks.apollo.models import Contact
        from backend.blocks.apollo.people import SearchPeopleBlock

        block = SearchPeopleBlock()
        fake_people = [Contact(id=str(i), first_name=f"Person{i}") for i in range(5)]
        accumulated: list[NodeExecutionStats] = []

        with (
            patch.object(
                SearchPeopleBlock,
                "search_people",
                new_callable=AsyncMock,
                return_value=fake_people,
            ),
            patch.object(
                block, "merge_stats", side_effect=lambda s: accumulated.append(s)
            ),
        ):
            input_data = SearchPeopleBlock.Input(
                credentials=APOLLO_CREDS_INPUT,  # type: ignore[arg-type]
            )
            async for _ in block.run(input_data, credentials=APOLLO_CREDS):
                pass

        assert len(accumulated) == 1
        assert accumulated[0].provider_cost == pytest.approx(5.0)
        assert accumulated[0].provider_cost_type == "items"

    @pytest.mark.asyncio
    async def test_empty_people_list_tracks_zero(self):
        """An empty people list results in provider_cost=0.0."""
        from backend.blocks.apollo._auth import TEST_CREDENTIALS as APOLLO_CREDS
        from backend.blocks.apollo._auth import (
            TEST_CREDENTIALS_INPUT as APOLLO_CREDS_INPUT,
        )
        from backend.blocks.apollo.people import SearchPeopleBlock

        block = SearchPeopleBlock()
        accumulated: list[NodeExecutionStats] = []

        with (
            patch.object(
                SearchPeopleBlock,
                "search_people",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch.object(
                block, "merge_stats", side_effect=lambda s: accumulated.append(s)
            ),
        ):
            input_data = SearchPeopleBlock.Input(
                credentials=APOLLO_CREDS_INPUT,  # type: ignore[arg-type]
            )
            async for _ in block.run(input_data, credentials=APOLLO_CREDS):
                pass

        assert len(accumulated) == 1
        assert accumulated[0].provider_cost == 0.0
        assert accumulated[0].provider_cost_type == "items"
