"""SDK `with_base_cost` prices reach every process, and only bill platform keys.

The executor and copilot executor never run `initialize_blocks()`; they only
load blocks (`get_block` -> `load_all_blocks`), which syncs provider costs. The
in-process tests start from that state: the block's BLOCK_COSTS entry removed,
then priced by the same `sync_all_provider_costs` call `load_all_blocks` makes.
They never clear or re-run the cached `load_all_blocks()`: in the full suite,
test modules define their own Block subclasses (e.g. `TestWebhookBlock`), so a
fresh load would pick those up and fail. The subprocess test covers the loader.
"""

import subprocess
import sys
import textwrap

import pytest
from pydantic import SecretStr

from backend.blocks._base import BlockCost, BlockCostType
from backend.blocks.agent_mail.inbox import AgentMailListInboxesBlock
from backend.blocks.baas.bots import BaasBotJoinMeetingBlock
from backend.blocks.dataforseo._config import dataforseo
from backend.blocks.dataforseo.keyword_suggestions import (
    DataForSeoKeywordSuggestionsBlock,
)
from backend.blocks.exa.search import ExaSearchBlock
from backend.data.block_cost_config import BLOCK_COSTS
from backend.data.model import (
    PLATFORM_FUNDED_COST_KEY,
    APIKeyCredentials,
    NodeExecutionStats,
    UserPasswordCredentials,
    is_refused_sdk_default,
    platform_funded_managed_id,
)
from backend.executor.utils import block_usage_cost
from backend.integrations.credentials_store import exa_credentials
from backend.sdk.cost_integration import (
    register_provider_costs_for_block,
    sync_all_provider_costs,
)

AGENT_MAIL_MANAGED_KEY = APIKeyCredentials(
    id=platform_funded_managed_id("agent_mail"),
    provider="agent_mail",
    api_key=SecretStr("pod-key"),
    title="AgentMail (managed by AutoGPT)",
    is_managed=True,
)
DATAFORSEO_PLATFORM_KEY = UserPasswordCredentials(
    id="dataforseo-default",
    provider="dataforseo",
    username=SecretStr("platform-user"),
    password=SecretStr("platform-password"),
    title="DataForSEO Credentials",
)


@pytest.fixture(autouse=True)
def restore_block_costs():
    saved = dict(BLOCK_COSTS)
    yield
    BLOCK_COSTS.clear()
    BLOCK_COSTS.update(saved)


def _credentials(cred_id: str, provider: str) -> dict:
    return {"credentials": {"id": cred_id, "provider": provider, "type": "api_key"}}


def _block_loaded_like_the_executor(block_class: type):
    """Price the block the way the executor's `load_all_blocks()` does, in a
    process that never ran initialize_blocks(): no BLOCK_COSTS entry yet, then
    the provider costs synced for the loaded blocks."""
    BLOCK_COSTS.pop(block_class, None)
    sync_all_provider_costs([block_class])
    return block_class()


def test_fresh_process_bills_sdk_price_like_the_executor():
    """The regression test that would have caught it: a new interpreter, like the
    executor and copilot executor processes, that loads blocks on demand and
    never runs initialize_blocks(). The pytest process can't show this, because
    the session test server already ran initialize_blocks() in it."""
    script = textwrap.dedent(
        f"""
        from backend.blocks import get_block
        from backend.data.model import NodeExecutionStats
        from backend.executor.utils import block_usage_cost

        block = get_block("{ExaSearchBlock().id}")
        stats = NodeExecutionStats(provider_cost=0.05, provider_cost_type="cost_usd")
        for cred_id in ("{exa_credentials.id}", "user-own-key"):
            credentials = {{"credentials": {{"id": cred_id, "provider": "exa"}}}}
            cost, _ = block_usage_cost(block, credentials, stats=stats)
            print("COST", cost)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )

    costs = [
        line.split()[1]
        for line in result.stdout.splitlines()
        if line.startswith("COST ")
    ]
    # Platform key: $0.05 * 150 credits/USD = 7.5, billed as 8; own key: free.
    assert costs == ["8", "0"]


def test_agent_mail_bills_its_run_price_on_the_managed_key():
    block = _block_loaded_like_the_executor(AgentMailListInboxesBlock)

    cost, _ = block_usage_cost(
        block, _credentials(AGENT_MAIL_MANAGED_KEY.id, "agent_mail")
    )

    assert cost == 1


def test_agent_mail_platform_marker_comes_from_the_id_not_the_input():
    block = _block_loaded_like_the_executor(AgentMailListInboxesBlock)
    managed = _credentials(AGENT_MAIL_MANAGED_KEY.id, "agent_mail")
    managed["credentials"][PLATFORM_FUNDED_COST_KEY] = False
    own = _credentials("user-own-key", "agent_mail")
    own["credentials"][PLATFORM_FUNDED_COST_KEY] = True

    assert block_usage_cost(block, managed)[0] == 1
    assert block_usage_cost(block, own)[0] == 0


def test_agent_mail_bills_the_managed_key_under_a_legacy_provider_string():
    block = _block_loaded_like_the_executor(AgentMailListInboxesBlock)

    cost, _ = block_usage_cost(
        block, _credentials(AGENT_MAIL_MANAGED_KEY.id, "ProviderName.AGENT_MAIL")
    )

    assert cost == 1


def test_agent_mail_has_no_shared_key_to_run_or_bill():
    block = _block_loaded_like_the_executor(AgentMailListInboxesBlock)

    cost, _ = block_usage_cost(block, _credentials("agent_mail-default", "agent_mail"))

    assert is_refused_sdk_default("agent_mail-default")
    assert cost == 0


def test_executor_lookup_bills_sdk_usd_price_on_the_platform_key():
    block = _block_loaded_like_the_executor(ExaSearchBlock)
    stats = NodeExecutionStats(provider_cost=0.05, provider_cost_type="cost_usd")

    cost, _ = block_usage_cost(
        block, _credentials(exa_credentials.id, "exa"), stats=stats
    )

    # $0.05 provider spend * 150 credits/USD = 7.5, billed as 8.
    assert cost == 8


def test_dataforseo_bills_its_reported_spend_at_the_platform_rate(monkeypatch):
    monkeypatch.setattr(dataforseo, "default_credentials", [DATAFORSEO_PLATFORM_KEY])
    block = _block_loaded_like_the_executor(DataForSeoKeywordSuggestionsBlock)
    stats = NodeExecutionStats(provider_cost=0.021, provider_cost_type="cost_usd")

    cost, _ = block_usage_cost(
        block, _credentials(DATAFORSEO_PLATFORM_KEY.id, "dataforseo"), stats=stats
    )

    # A $0.021 DataForSEO task * 150 credits/USD = 3.15, billed as 4 credits.
    assert cost == 4


def test_own_key_runs_stay_free():
    mail = _block_loaded_like_the_executor(AgentMailListInboxesBlock)
    exa = _block_loaded_like_the_executor(ExaSearchBlock)
    stats = NodeExecutionStats(provider_cost=0.05, provider_cost_type="cost_usd")

    own_mail, _ = block_usage_cost(mail, _credentials("user-own-key", "agent_mail"))
    own_exa, _ = block_usage_cost(exa, _credentials("user-own-key", "exa"), stats=stats)

    assert own_mail == 0
    assert own_exa == 0


def test_provider_without_a_platform_key_bills_nothing(monkeypatch):
    monkeypatch.setattr(dataforseo, "default_credentials", [])
    block = _block_loaded_like_the_executor(DataForSeoKeywordSuggestionsBlock)
    stats = NodeExecutionStats(provider_cost=0.021, provider_cost_type="cost_usd")

    cost, _ = block_usage_cost(
        block, _credentials("user-own-key", "dataforseo"), stats=stats
    )

    assert cost == 0


def test_a_zero_amount_tokens_price_still_needs_the_platform_key(monkeypatch):
    """TOKENS prices from the token rate table, not cost_amount, so a zero
    amount doesn't make it free. Only a free cost stays unfiltered."""
    monkeypatch.setattr(dataforseo, "default_credentials", [DATAFORSEO_PLATFORM_KEY])
    monkeypatch.setattr(
        dataforseo,
        "base_costs",
        [
            BlockCost(cost_amount=0),
            BlockCost(cost_amount=0, cost_type=BlockCostType.TOKENS),
        ],
    )

    _block_loaded_like_the_executor(DataForSeoKeywordSuggestionsBlock)

    assert [c.cost_filter for c in BLOCK_COSTS[DataForSeoKeywordSuggestionsBlock]] == [
        {},
        {"credentials": {"id": DATAFORSEO_PLATFORM_KEY.id}},
    ]


def test_existing_block_costs_entries_are_left_alone():
    before = BLOCK_COSTS[BaasBotJoinMeetingBlock]

    register_provider_costs_for_block(BaasBotJoinMeetingBlock)

    assert BLOCK_COSTS[BaasBotJoinMeetingBlock] is before


def test_syncing_twice_does_not_add_costs():
    BLOCK_COSTS.pop(AgentMailListInboxesBlock, None)

    sync_all_provider_costs([AgentMailListInboxesBlock])
    first = list(BLOCK_COSTS[AgentMailListInboxesBlock])
    sync_all_provider_costs([AgentMailListInboxesBlock])

    assert BLOCK_COSTS[AgentMailListInboxesBlock] == first
    assert len(first) == 1
