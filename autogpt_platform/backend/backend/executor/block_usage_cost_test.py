"""Tests for the dynamic-pricing branches of block_usage_cost."""

import ast
import math
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.blocks._base import BlockCost, BlockCostType
from backend.blocks.ayrshare.post_to_youtube import PostToYouTubeBlock
from backend.blocks.code_executor import (
    ExecuteCodeBlock,
    ExecuteCodeStepBlock,
    InstantiateCodeSandboxBlock,
)
from backend.blocks.exa.search import ExaSearchBlock
from backend.blocks.fal.ai_video_generator import AIVideoGeneratorBlock
from backend.blocks.jina.search import ExtractWebsiteContentBlock, SearchTheWebBlock
from backend.blocks.llm import DEFAULT_LLM_MODEL, AITextGeneratorBlock, LLMModel
from backend.data.block_cost_config import (
    BLOCK_COSTS,
    MODEL_COST,
    TOKEN_COST,
    TokenRate,
)
from backend.data.llm_registry import llm_models
from backend.data.model import NodeExecutionStats
from backend.executor import utils as executor_utils
from backend.executor.billing import _block_has_paid_cost_entry
from backend.executor.utils import block_usage_cost
from backend.integrations.credentials_store import (
    anthropic_credentials,
    e2b_credentials,
    exa_credentials,
    fal_credentials,
    jina_credentials,
    openai_credentials,
)
from backend.sdk.cost_integration import register_provider_costs_for_block


@pytest.fixture
def registered_exa_costs():
    with patch.dict(BLOCK_COSTS):
        BLOCK_COSTS.pop(ExaSearchBlock, None)
        register_provider_costs_for_block(ExaSearchBlock)
        yield


@pytest.fixture(autouse=True)
def _stub_preflight_estimate(monkeypatch):
    """Force `get_preflight_estimate` to return 0 in this module's tests so
    they don't accidentally couple to a populated
    `block_preflight_estimates.json` once the admin export tool seeds it.
    Tests that want a non-zero estimate (e.g.
    `test_preflight_uses_historical_estimate_for_dynamic_cost_types`) override
    this themselves on `executor_utils.get_preflight_estimate`.
    """
    monkeypatch.setattr(executor_utils, "get_preflight_estimate", lambda _bid: 0)


@pytest.fixture
def tmp_block_costs_override(monkeypatch):
    """Swap out BLOCK_COSTS[SearchTheWebBlock] for the duration of a test."""
    return lambda costs: monkeypatch.setitem(BLOCK_COSTS, SearchTheWebBlock, costs)


def test_second_cost_type_uses_stats_walltime_with_divisor(tmp_block_costs_override):
    tmp_block_costs_override(
        [BlockCost(cost_amount=1, cost_type=BlockCostType.SECOND, cost_divisor=10)]
    )
    block = SearchTheWebBlock()
    # 25 seconds of walltime / 10 sec-per-credit = ceil(2.5) = 3 credits.
    stats = NodeExecutionStats(walltime=25.0)
    cost, _ = block_usage_cost(block, {}, stats=stats)
    assert cost == 3


def test_second_cost_type_sub_divisor_bills_one_credit(tmp_block_costs_override):
    """Sub-divisor walltime still bills 1cr — no 0-credit leak."""
    tmp_block_costs_override(
        [BlockCost(cost_amount=1, cost_type=BlockCostType.SECOND, cost_divisor=3)]
    )
    block = SearchTheWebBlock()
    # 1s walltime on a 1cr/3s block → ceil(1/3) * 1 = 1 credit.
    stats = NodeExecutionStats(walltime=1.0)
    cost, _ = block_usage_cost(block, {}, stats=stats)
    assert cost == 1


def test_second_cost_type_returns_zero_without_stats_or_runtime(
    tmp_block_costs_override,
):
    tmp_block_costs_override([BlockCost(cost_amount=1, cost_type=BlockCostType.SECOND)])
    block = SearchTheWebBlock()
    # Pre-flight: no stats, no run_time. SECOND should return 0 credits.
    cost, _ = block_usage_cost(block, {})
    assert cost == 0


def test_items_cost_type_multiplies_provider_cost(tmp_block_costs_override):
    tmp_block_costs_override(
        [BlockCost(cost_amount=1, cost_type=BlockCostType.ITEMS, cost_divisor=10)]
    )
    block = SearchTheWebBlock()
    stats = NodeExecutionStats(provider_cost=45, provider_cost_type="items")
    # 45 items / 10 = ceil(4.5) = 5 credits.
    cost, _ = block_usage_cost(block, {}, stats=stats)
    assert cost == 5


def test_items_cost_type_sub_divisor_bills_one_credit(tmp_block_costs_override):
    """A single item under cost_divisor=2 still bills 1cr — no 0-credit leak."""
    tmp_block_costs_override(
        [BlockCost(cost_amount=1, cost_type=BlockCostType.ITEMS, cost_divisor=2)]
    )
    block = SearchTheWebBlock()
    # Apollo SearchOrganizationsBlock shape: 1 result returned on a 1cr/2-item
    # block should bill ceil(1/2) * 1 = 1 credit (floor division would bill 0).
    stats = NodeExecutionStats(provider_cost=1, provider_cost_type="items")
    cost, _ = block_usage_cost(block, {}, stats=stats)
    assert cost == 1


def test_items_cost_type_ignores_non_items_provider_cost(tmp_block_costs_override):
    tmp_block_costs_override([BlockCost(cost_amount=1, cost_type=BlockCostType.ITEMS)])
    block = SearchTheWebBlock()
    # provider_cost is USD, not item count — don't misread as items.
    stats = NodeExecutionStats(provider_cost=0.05, provider_cost_type="cost_usd")
    cost, _ = block_usage_cost(block, {}, stats=stats)
    assert cost == 0


def test_cost_usd_ceils_multiplier(tmp_block_costs_override):
    tmp_block_costs_override(
        # 100 credits per USD, so $0.023 provider spend → 3 credits (ceil of 2.3).
        [BlockCost(cost_amount=100, cost_type=BlockCostType.COST_USD)]
    )
    block = SearchTheWebBlock()
    stats = NodeExecutionStats(provider_cost=0.023, provider_cost_type="cost_usd")
    cost, _ = block_usage_cost(block, {}, stats=stats)
    assert cost == 3


def test_cost_usd_zero_when_no_stats(tmp_block_costs_override):
    tmp_block_costs_override(
        [BlockCost(cost_amount=100, cost_type=BlockCostType.COST_USD)]
    )
    block = SearchTheWebBlock()
    cost, _ = block_usage_cost(block, {})
    assert cost == 0


def test_preflight_estimate_disabled_for_no_reconciliation_callers(
    tmp_block_costs_override, monkeypatch
):
    """The direct block-execute API endpoints bypass the executor manager
    and have no post-flight reconciliation. They MUST pass
    `use_preflight_estimate=False` so dynamic-cost blocks return 0 instead
    of locking in an unreconciled estimate as the final charge.
    """
    tmp_block_costs_override(
        [BlockCost(cost_amount=1, cost_type=BlockCostType.SECOND, cost_divisor=10)]
    )
    block = SearchTheWebBlock()

    # Even with a registered estimate, the flag wins.
    monkeypatch.setattr(executor_utils, "get_preflight_estimate", lambda _bid: 999)

    cost, _ = block_usage_cost(block, {}, use_preflight_estimate=False)
    assert cost == 0

    # Sanity: with the flag default-True, the same call returns the estimate.
    cost_with_estimate, _ = block_usage_cost(block, {})
    assert cost_with_estimate == 999


def test_preflight_uses_historical_estimate_for_dynamic_cost_types(
    tmp_block_costs_override, monkeypatch
):
    """When stats is None and run_time=0, dynamic-cost branches return the
    registered historical-average estimate instead of 0 — so the post-flight
    reconciliation only settles a small delta and a billing leak is bounded
    by that delta rather than the full execution cost."""
    tmp_block_costs_override(
        [BlockCost(cost_amount=1, cost_type=BlockCostType.SECOND, cost_divisor=10)]
    )
    block = SearchTheWebBlock()

    monkeypatch.setattr(
        executor_utils,
        "get_preflight_estimate",
        lambda block_id: 5 if block_id == block.id else 0,
    )

    cost, _ = block_usage_cost(block, {})
    assert cost == 5


def test_cost_usd_ignores_non_usd_provider_cost(tmp_block_costs_override):
    """provider_cost_type='items' should not be mistaken for dollars."""
    tmp_block_costs_override(
        [BlockCost(cost_amount=100, cost_type=BlockCostType.COST_USD)]
    )
    block = SearchTheWebBlock()
    # An items-typed provider_cost of 45 would otherwise be billed as $45.
    stats = NodeExecutionStats(provider_cost=45, provider_cost_type="items")
    cost, _ = block_usage_cost(block, {}, stats=stats)
    assert cost == 0


def test_tokens_cost_type_uses_token_rate_table(tmp_block_costs_override, monkeypatch):
    tmp_block_costs_override([BlockCost(cost_amount=0, cost_type=BlockCostType.TOKENS)])
    # Override TOKEN_COST for a predictable rate: 1000 credits/1M input,
    # 2000 credits/1M output.
    monkeypatch.setitem(
        TOKEN_COST,
        LLMModel.GPT4O_MINI,
        TokenRate(input=1000, output=2000),
    )
    block = SearchTheWebBlock()
    stats = NodeExecutionStats(
        input_token_count=500_000,
        output_token_count=250_000,
    )
    cost, _ = block_usage_cost(
        block,
        {"model": LLMModel.GPT4O_MINI.value},
        stats=stats,
    )
    # 0.5 * 1000 + 0.25 * 2000 = 500 + 500 = 1000 credits.
    assert cost == 1000


def test_tokens_falls_back_to_flat_model_cost_when_rate_missing(
    tmp_block_costs_override,
):
    tmp_block_costs_override([BlockCost(cost_amount=0, cost_type=BlockCostType.TOKENS)])
    block = SearchTheWebBlock()
    # Ollama models aren't in TOKEN_COST but are in MODEL_COST.
    ollama_model = LLMModel.OLLAMA_LLAMA3_2
    expected = MODEL_COST[ollama_model]
    cost, _ = block_usage_cost(
        block,
        {"model": ollama_model.value},
        stats=NodeExecutionStats(input_token_count=10_000, output_token_count=10_000),
    )
    assert cost == expected


def test_ollama_zero_cost_entry_matches_without_credentials():
    ollama_model = LLMModel.OLLAMA_LLAMA3_2

    cost, matching_filter = block_usage_cost(
        AITextGeneratorBlock(),
        {"model": ollama_model.value},
    )

    assert cost == 0
    assert matching_filter == {"model": ollama_model}


def test_e2b_sandbox_blocks_bill_one_credit_per_ten_seconds():
    """End-to-end: E2B blocks use the real SECOND/divisor=10 BlockCost entry."""
    creds = {
        "credentials": {
            "id": e2b_credentials.id,
            "provider": e2b_credentials.provider,
            "type": e2b_credentials.type,
        }
    }
    for block_cls in (
        ExecuteCodeBlock,
        InstantiateCodeSandboxBlock,
        ExecuteCodeStepBlock,
    ):
        # 45s walltime → ceil(45/10) = 5 credits.
        stats = NodeExecutionStats(walltime=45.0)
        cost, _ = block_usage_cost(block_cls(), creds, stats=stats)
        assert cost == 5, f"{block_cls.__name__} expected 5 credits, got {cost}"
        # Pre-flight (no stats) → 0.
        cost, _ = block_usage_cost(block_cls(), creds)
        assert cost == 0


def test_fal_video_block_bills_fifteen_credits_per_second():
    block = AIVideoGeneratorBlock()
    creds = {
        "credentials": {
            "id": fal_credentials.id,
            "provider": fal_credentials.provider,
            "type": fal_credentials.type,
        }
    }
    # 8s clip → 15 * 8 = 120 credits.
    cost, _ = block_usage_cost(block, creds, stats=NodeExecutionStats(walltime=8.0))
    assert cost == 120
    # Pre-flight → 0.
    cost, _ = block_usage_cost(block, creds)
    assert cost == 0


def test_exa_blocks_bill_cost_usd_via_sdk_config(registered_exa_costs):
    """End-to-end: Exa's ProviderBuilder.with_base_cost(150, COST_USD) is live."""
    block = ExaSearchBlock()
    creds = {
        "credentials": {
            "id": exa_credentials.id,
            "provider": exa_credentials.provider,
            "type": exa_credentials.type,
        }
    }
    # $0.05 provider spend * 150 credits/USD = 7.5, billed as 8 credits.
    stats = NodeExecutionStats(provider_cost=0.05, provider_cost_type="cost_usd")
    cost, _ = block_usage_cost(block, creds, stats=stats)
    assert cost == 8
    # Pre-flight: unknown cost → 0.
    cost, _ = block_usage_cost(block, creds)
    assert cost == 0


def test_llm_block_charges_per_token_post_flight():
    """AITextGeneratorBlock with Claude 4.6 Sonnet bills by real token counts."""
    block = AITextGeneratorBlock()
    input_data = {
        "model": LLMModel.CLAUDE_4_6_SONNET,
        "credentials": {
            "id": anthropic_credentials.id,
            "provider": anthropic_credentials.provider,
            "type": anthropic_credentials.type,
        },
    }
    rate = TOKEN_COST[LLMModel.CLAUDE_4_6_SONNET]
    stats = NodeExecutionStats(
        input_token_count=200_000,
        output_token_count=50_000,
        cache_read_token_count=100_000,
    )
    expected = math.ceil(
        (200_000 * rate.input + 50_000 * rate.output + 100_000 * rate.cache_read)
        / 1_000_000
    )
    cost, _ = block_usage_cost(block, input_data, stats=stats)
    assert cost == expected


def test_llm_block_pre_flight_falls_back_to_model_cost():
    """Pre-flight charge of an LLM block uses the flat MODEL_COST floor."""
    block = AITextGeneratorBlock()
    cost, _ = block_usage_cost(
        block,
        {
            "model": LLMModel.GPT6_1_SOL,
            "credentials": {
                "id": openai_credentials.id,
                "provider": openai_credentials.provider,
                "type": openai_credentials.type,
            },
        },
    )
    assert cost == MODEL_COST[LLMModel.GPT6_1_SOL]


@pytest.mark.parametrize("schema_default_input", [{}, {"model": None}])
def test_llm_block_price_includes_schema_defaults(schema_default_input: dict):
    """Pricing fills schema defaults the way the block run does."""
    block = AITextGeneratorBlock()
    credentials = {
        "id": openai_credentials.id,
        "provider": openai_credentials.provider,
        "type": openai_credentials.type,
    }
    explicit = {"model": DEFAULT_LLM_MODEL.value, "credentials": credentials}
    implicit = {**schema_default_input, "credentials": credentials}
    stats = NodeExecutionStats(input_token_count=200_000, output_token_count=50_000)

    assert block_usage_cost(block, implicit)[0] == MODEL_COST[DEFAULT_LLM_MODEL] > 0
    assert block_usage_cost(block, implicit) == block_usage_cost(block, explicit)
    assert block_usage_cost(block, implicit, stats=stats) == block_usage_cost(
        block, explicit, stats=stats
    )


@pytest.mark.parametrize(
    ("day", "runs"),
    [
        # Still served the day before its shutdown date: its own price.
        (date(2026, 12, 10), LLMModel.GPT5_PRO),
        # From that day on it runs, and is priced as, its fallback.
        (date(2026, 12, 11), LLMModel.GPT6_1_SOL),
    ],
    ids=["before-shutdown", "on-the-day"],
)
def test_a_switched_off_model_is_charged_as_the_model_it_runs(
    day: date, runs: LLMModel, monkeypatch: pytest.MonkeyPatch
):
    """A block given a model the catalog has switched off runs the model's
    catalog fallback from its shutdown date (`Block.with_retired_models_replaced`),
    and is charged for the model it runs: gpt-5-pro, then gpt-6.1-sol."""
    monkeypatch.setattr(llm_models, "_today", lambda: day)
    block = AITextGeneratorBlock()
    credentials = {
        "id": openai_credentials.id,
        "provider": openai_credentials.provider,
        "type": openai_credentials.type,
    }
    stored = {"model": LLMModel.GPT5_PRO.value, "credentials": credentials}
    priced_as = {"model": runs.value, "credentials": credentials}
    stats = NodeExecutionStats(input_token_count=1_000_000, output_token_count=0)

    cost, matched = block_usage_cost(block, stored, stats=stats)

    assert matched["model"] == runs
    assert cost == math.ceil(TOKEN_COST[runs].input)
    assert block_usage_cost(block, stored) == block_usage_cost(block, priced_as)
    assert executor_utils._with_schema_defaults(block, stored, None)["model"] == runs


def _credentials(credentials) -> dict:
    return {
        "id": credentials.id,
        "provider": credentials.provider,
        "type": credentials.type,
    }


@pytest.mark.parametrize(
    ("block", "as_it_runs", "as_sent"),
    [
        pytest.param(
            AITextGeneratorBlock(),
            {
                "model": LLMModel.CLAUDE_4_5_OPUS.value,
                "credentials": _credentials(anthropic_credentials),
            },
            {"model": "anthropic/claude-opus-4-5"},
            id="provider-prefixed-model",
        ),
        pytest.param(
            ExtractWebsiteContentBlock(),
            {
                "url": "https://example.com",
                "raw_content": False,
                "credentials": _credentials(jina_credentials),
            },
            {"raw_content": "false"},
            id="flag-as-string",
        ),
    ],
)
def test_price_follows_the_value_the_run_coerces(block, as_it_runs, as_sent):
    """Input not yet coerced to the block's types, as an Orchestrator tool
    call's arguments reach the charge, is priced as the value the block runs."""
    sent = {**as_it_runs, **as_sent}
    stats = NodeExecutionStats(input_token_count=200_000, output_token_count=50_000)

    assert block_usage_cost(block, sent)[0] > 0
    assert block_usage_cost(block, sent) == block_usage_cost(block, as_it_runs)
    assert block_usage_cost(block, sent, stats=stats) == block_usage_cost(
        block, as_it_runs, stats=stats
    )
    assert _block_has_paid_cost_entry(block, sent)
    assert sent == {**as_it_runs, **as_sent}


def test_pricing_leaves_a_value_the_run_refuses_unpriced():
    """A value no coercion can turn into the block's type fails the run, not
    the pricing."""
    refused = {"model": "no-such-model", "prompt": "hi"}

    assert block_usage_cost(AITextGeneratorBlock(), refused) == (0, {})


def test_ayrshare_price_tier_follows_the_is_video_default():
    """YouTube's `is_video` default picks its price tier."""
    block = PostToYouTubeBlock()
    assert block_usage_cost(block, {}) == block_usage_cost(block, {"is_video": True})


def test_every_cost_filter_match_uses_the_run_input():
    """Each function that matches cost filters builds the input the block runs
    on with `_with_schema_defaults`, including any added after these two."""
    backend_root = Path(executor_utils.__file__).parents[1]
    callers: dict[str, set[str | None]] = {}
    for path in backend_root.rglob("*.py"):
        source = path.read_text()
        if path.name.endswith("_test.py") or "_is_cost_filter_match" not in source:
            continue
        for func in ast.walk(ast.parse(source)):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            called = {
                _called_name(node)
                for node in ast.walk(func)
                if isinstance(node, ast.Call)
            }
            if (
                "_is_cost_filter_match" in called
                and func.name != "_is_cost_filter_match"
            ):
                callers[f"{path.relative_to(backend_root)}::{func.name}"] = called

    assert {
        "executor/utils.py::block_usage_cost",
        "executor/billing.py::_block_has_paid_cost_entry",
    } <= callers.keys()
    assert [
        name
        for name, called in callers.items()
        if "_with_schema_defaults" not in called
    ] == []


def test_every_executor_reader_of_an_input_model_reads_the_model_that_runs():
    """A function in the executor that reads an input's `model` reads the model
    the block runs: it goes through `_with_schema_defaults` or
    `with_retired_models_replaced`. The cost log, the dry-run usage record and
    the stop estimate read it today, as must any reader added after them."""
    executor_root = Path(executor_utils.__file__).parent
    readers: dict[str, set[str | None]] = {}
    for path in executor_root.rglob("*.py"):
        if path.name.endswith("_test.py"):
            continue
        for func in ast.walk(ast.parse(path.read_text())):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = [node for node in ast.walk(func) if isinstance(node, ast.Call)]
            if any(_reads_model(call) for call in calls):
                readers[f"{path.relative_to(executor_root)}::{func.name}"] = {
                    _called_name(call) for call in calls
                }

    assert {
        "cost_tracking.py::log_system_credential_cost",
        "manager.py::execute_node",
        "billing.py::_with_stop_estimate",
    } <= readers.keys()
    assert [
        name
        for name, called in readers.items()
        if not called & {"_with_schema_defaults", "with_retired_models_replaced"}
    ] == []


# The lookups made before any run has started, which may price or name the
# model by today's date: the model a run starting now would run.
BEFORE_A_RUN = {
    "blocks/_base.py::_execute": "the swap itself: the run starts now",
    "executor/billing.py::resolve_block_cost": "the pre-flight charge",
    "executor/billing.py::_block_has_paid_cost_entry": "the pre-flight balance check",
    "executor/billing.py::usage_hold": "the hold taken before a run",
    "executor/billing.py::redispatch_reconcile_baseline": (
        "a run parked for review, which ran no model"
    ),
    "executor/utils.py::charge_for_direct_block_execution": (
        "charged just before the block starts"
    ),
    "copilot/tools/helpers.py::_pre_run_charge": "AutoPilot's debit before a run",
    "copilot/gate/subject.py::block_subject": "an approval card's price",
    "copilot/gate/subject.py::graph_cost_credits": "an approval card's price",
}


def test_a_lookup_after_a_run_reads_the_model_the_run_recorded():
    """Once a run has started, its price and its model's name come from the
    model it recorded (`NodeExecutionStats.resolved_model`), never from
    today's date: a node still running at midnight on its model's shutdown
    date is billed for the model it ran. So every call of
    `block_usage_cost` (`stats=` or `pre_flight_resolved_model=`),
    `_with_schema_defaults` and `with_retired_models_replaced` (their
    `resolved_model`) passes that record, never a literal None or an empty
    `NodeExecutionStats()`, except the calls in `BEFORE_A_RUN`."""
    backend_root = Path(executor_utils.__file__).parents[1]
    callers: set[str] = set()
    unrecorded: list[str] = []
    for path in backend_root.rglob("*.py"):
        source = path.read_text()
        if path.name.endswith("_test.py") or not any(
            lookup in source for lookup in _MODEL_LOOKUPS
        ):
            continue
        for func in ast.walk(ast.parse(source)):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            where = f"{path.relative_to(backend_root)}::{func.name}"
            for call in (node for node in ast.walk(func) if isinstance(node, ast.Call)):
                is_lookup, record = _record_passed(call)
                if not is_lookup:
                    continue
                callers.add(where)
                if where not in BEFORE_A_RUN and not _names_a_record(record):
                    unrecorded.append(f"{where}:{call.lineno}")

    assert BEFORE_A_RUN.keys() <= callers
    assert unrecorded == []


# Each lookup that takes a run's record, with the position of its
# `resolved_model` argument (`block_usage_cost` takes it by keyword).
_MODEL_LOOKUPS = {
    "block_usage_cost": None,
    "_with_schema_defaults": 2,
    "with_retired_models_replaced": 1,
}


def _record_passed(call: ast.Call) -> tuple[bool, ast.expr | None]:
    """Whether *call* is one of the model lookups, and the argument by which
    it passes a run's record (None when it passes none)."""
    name = _called_name(call) or ""
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    if name == "block_usage_cost":
        return True, keywords.get("stats") or keywords.get("pre_flight_resolved_model")
    position = _MODEL_LOOKUPS.get(name)
    if position is None:
        return False, None
    if len(call.args) > position:
        return True, call.args[position]
    return True, keywords.get("resolved_model")


def _names_a_record(argument: ast.expr | None) -> bool:
    if argument is None:
        return False
    if isinstance(argument, ast.Constant) and argument.value is None:
        return False
    return not (
        isinstance(argument, ast.Call)
        and _called_name(argument) == "NodeExecutionStats"
        and all(keyword.arg != "resolved_model" for keyword in argument.keywords)
    )


def _reads_model(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "get"
        and bool(call.args)
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "model"
    )


def _called_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def test_run_cost_type_remains_unchanged(tmp_block_costs_override):
    tmp_block_costs_override([BlockCost(cost_amount=7, cost_type=BlockCostType.RUN)])
    block = SearchTheWebBlock()
    # Stats shouldn't affect RUN charge.
    stats = NodeExecutionStats(walltime=999, input_token_count=999)
    cost, _ = block_usage_cost(block, {}, stats=stats)
    assert cost == 7
