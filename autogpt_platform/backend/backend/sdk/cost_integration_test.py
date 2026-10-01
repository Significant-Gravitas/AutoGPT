"""SDK `with_base_cost` prices reach every process, and only bill platform keys.

The executor and copilot executor never run `initialize_blocks()`; they only
load blocks (`get_block` -> `load_all_blocks`). These tests start from that
state: the block's BLOCK_COSTS entry removed and the block cache cleared.
"""

import subprocess
import sys
import textwrap

import pytest
from pydantic import SecretStr

from backend.blocks import load_all_blocks
from backend.blocks.agent_mail._config import agent_mail
from backend.blocks.agent_mail.inbox import AgentMailListInboxesBlock
from backend.blocks.baas.bots import BaasBotJoinMeetingBlock
from backend.blocks.exa.search import ExaSearchBlock
from backend.data.block_cost_config import BLOCK_COSTS
from backend.data.model import APIKeyCredentials, NodeExecutionStats
from backend.executor.utils import block_usage_cost
from backend.integrations.credentials_store import exa_credentials
from backend.sdk.cost_integration import (
    register_provider_costs_for_block,
    sync_all_provider_costs,
)

AGENT_MAIL_PLATFORM_KEY = APIKeyCredentials(
    id="agent_mail-default",
    provider="agent_mail",
    api_key=SecretStr("platform-key"),
    title="AgentMail API Key",
)


@pytest.fixture(autouse=True)
def restore_block_costs():
    saved = dict(BLOCK_COSTS)
    yield
    BLOCK_COSTS.clear()
    BLOCK_COSTS.update(saved)
    load_all_blocks.cache_clear()


def _credentials(cred_id: str, provider: str) -> dict:
    return {"credentials": {"id": cred_id, "provider": provider, "type": "api_key"}}


def _block_loaded_like_the_executor(block_class: type):
    """Fetch the block the way the executor does, in a process that never ran
    initialize_blocks(): no BLOCK_COSTS entry yet, blocks loaded on demand."""
    BLOCK_COSTS.pop(block_class, None)
    load_all_blocks.cache_clear()
    blocks = load_all_blocks()
    return blocks[block_class().id]()


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
    # Platform key: $0.05 * 100 credits/USD; own key: free.
    assert costs == ["5", "0"]


def test_executor_lookup_bills_sdk_run_price_on_the_platform_key(monkeypatch):
    monkeypatch.setattr(agent_mail, "default_credentials", [AGENT_MAIL_PLATFORM_KEY])
    block = _block_loaded_like_the_executor(AgentMailListInboxesBlock)

    cost, _ = block_usage_cost(
        block, _credentials(AGENT_MAIL_PLATFORM_KEY.id, "agent_mail")
    )

    assert cost == 1


def test_executor_lookup_bills_sdk_usd_price_on_the_platform_key():
    block = _block_loaded_like_the_executor(ExaSearchBlock)
    stats = NodeExecutionStats(provider_cost=0.05, provider_cost_type="cost_usd")

    cost, _ = block_usage_cost(
        block, _credentials(exa_credentials.id, "exa"), stats=stats
    )

    # $0.05 provider spend * 100 credits/USD.
    assert cost == 5


def test_own_key_runs_stay_free(monkeypatch):
    monkeypatch.setattr(agent_mail, "default_credentials", [AGENT_MAIL_PLATFORM_KEY])
    mail = _block_loaded_like_the_executor(AgentMailListInboxesBlock)
    exa = _block_loaded_like_the_executor(ExaSearchBlock)
    stats = NodeExecutionStats(provider_cost=0.05, provider_cost_type="cost_usd")

    own_mail, _ = block_usage_cost(mail, _credentials("user-own-key", "agent_mail"))
    own_exa, _ = block_usage_cost(exa, _credentials("user-own-key", "exa"), stats=stats)

    assert own_mail == 0
    assert own_exa == 0


def test_provider_without_a_platform_key_bills_nothing(monkeypatch):
    monkeypatch.setattr(agent_mail, "default_credentials", [])
    block = _block_loaded_like_the_executor(AgentMailListInboxesBlock)

    cost, _ = block_usage_cost(block, _credentials("user-own-key", "agent_mail"))

    assert cost == 0


def test_existing_block_costs_entries_are_left_alone():
    before = BLOCK_COSTS[BaasBotJoinMeetingBlock]

    register_provider_costs_for_block(BaasBotJoinMeetingBlock)

    assert BLOCK_COSTS[BaasBotJoinMeetingBlock] is before


def test_syncing_twice_does_not_add_costs(monkeypatch):
    monkeypatch.setattr(agent_mail, "default_credentials", [AGENT_MAIL_PLATFORM_KEY])
    BLOCK_COSTS.pop(AgentMailListInboxesBlock, None)

    sync_all_provider_costs([AgentMailListInboxesBlock])
    first = list(BLOCK_COSTS[AgentMailListInboxesBlock])
    sync_all_provider_costs([AgentMailListInboxesBlock])

    assert BLOCK_COSTS[AgentMailListInboxesBlock] == first
    assert len(first) == 1
