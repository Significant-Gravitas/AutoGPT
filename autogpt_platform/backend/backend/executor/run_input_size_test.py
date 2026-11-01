from typing import cast
from unittest.mock import AsyncMock

import pytest
from pytest_mock import MockerFixture

from backend.blocks.basic import StoreValueBlock
from backend.blocks.io import AgentInputBlock
from backend.data.graph import Graph, GraphModel, Node, make_graph_model
from backend.data.model import CredentialsFieldInfo, CredentialsMetaInput
from backend.executor import utils
from backend.integrations.providers import ProviderName
from backend.util.exceptions import InvalidInputError


def _inputs_named_x(count: int) -> GraphModel:
    return make_graph_model(
        Graph(
            name="Inputs",
            description="",
            nodes=[
                Node(
                    id=f"input-{i}",
                    block_id=AgentInputBlock().id,
                    input_default={"name": "x"},
                )
                for i in range(count)
            ],
        ),
        user_id="owner",
    )


@pytest.fixture
def construct(mocker: MockerFixture) -> AsyncMock:
    mocker.patch.object(utils, "prisma").is_connected.return_value = True
    mocker.patch.object(
        utils.graph_db, "validate_graph_execution_permissions", AsyncMock()
    )
    return mocker.patch.object(
        utils,
        "_construct_starting_node_execution_input",
        AsyncMock(return_value=([], set())),
    )


async def _validate(
    mocker: MockerFixture, graph: GraphModel, graph_inputs: dict, **kwargs
) -> None:
    mocker.patch.object(utils.graph_db, "get_graph", AsyncMock(return_value=graph))
    await utils.validate_and_construct_node_execution_input(
        graph_id=graph.id, user_id="owner", graph_inputs=graph_inputs, **kwargs
    )


@pytest.mark.asyncio
async def test_a_run_at_the_input_budget_is_accepted(
    mocker: MockerFixture, construct: AsyncMock
):
    # The stored inputs {"x":"aaa…"} and seven block copies of the value, with
    # the empty overrides and credentials: as close to the budget as eighths get.
    value = "a" * (utils.MAX_RUN_INPUT_BYTES // 8 - 4)

    await _validate(mocker, _inputs_named_x(7), {"x": value})

    construct.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_run_over_the_input_budget_is_refused_before_it_is_built(
    mocker: MockerFixture, construct: AsyncMock
):
    measure = mocker.spy(utils, "json_dumps")
    value = "a" * (utils.MAX_RUN_INPUT_BYTES // 8 - 1)

    with pytest.raises(InvalidInputError, match="too large"):
        await _validate(mocker, _inputs_named_x(7), {"x": value})

    construct.assert_not_awaited()
    assert [call.args[0] for call in measure.call_args_list] == ["x", value]


@pytest.mark.asyncio
async def test_graph_inputs_without_an_input_block_count_toward_the_input_budget(
    mocker: MockerFixture, construct: AsyncMock
):
    big = "a" * (utils.MAX_RUN_INPUT_BYTES // 2)

    with pytest.raises(InvalidInputError, match="too large"):
        await _validate(mocker, _inputs_named_x(1), {"y": big, "z": big})

    construct.assert_not_awaited()


@pytest.mark.asyncio
async def test_stored_credentials_count_toward_the_input_budget(
    mocker: MockerFixture, construct: AsyncMock
):
    mocker.patch.object(
        utils, "make_node_credentials_input_map", mocker.MagicMock(return_value={})
    )
    credentials = CredentialsMetaInput(
        id="cred-1",
        title="t" * utils.MAX_RUN_INPUT_BYTES,
        provider=ProviderName("github"),
        type="api_key",
    )

    with pytest.raises(InvalidInputError, match="too large"):
        await _validate(
            mocker,
            _inputs_named_x(1),
            {},
            graph_credentials_inputs={"github": credentials},
        )

    construct.assert_not_awaited()


@pytest.mark.asyncio
async def test_node_input_overrides_count_toward_the_input_budget(
    mocker: MockerFixture, construct: AsyncMock
):
    big = "a" * (utils.MAX_RUN_INPUT_BYTES // 2)

    with pytest.raises(InvalidInputError, match="too large"):
        await _validate(
            mocker,
            _inputs_named_x(1),
            {},
            nodes_input_masks={"input-0": {"value": big}, "other": {"value": big}},
        )

    construct.assert_not_awaited()


@pytest.mark.asyncio
async def test_each_starting_node_copy_of_its_overrides_counts_toward_the_budget(
    mocker: MockerFixture, construct: AsyncMock
):
    # Stored once these come to about three fifths of the budget; each
    # starting node's copy of its own overrides takes that over.
    value = "a" * (utils.MAX_RUN_INPUT_BYTES // 5)
    nodes_input_masks = {f"input-{i}": {"value": value} for i in range(3)}

    with pytest.raises(InvalidInputError, match="too large"):
        await _validate(
            mocker, _inputs_named_x(3), {}, nodes_input_masks=nodes_input_masks
        )

    construct.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_non_input_starting_node_copy_of_its_overrides_counts(
    mocker: MockerFixture, construct: AsyncMock
):
    # Stored once the override is three fifths of the budget; the starting
    # node's copy of it takes that over.
    graph = make_graph_model(
        Graph(
            name="Store",
            description="",
            nodes=[Node(id="store-0", block_id=StoreValueBlock().id)],
        ),
        user_id="owner",
    )
    value = "a" * (utils.MAX_RUN_INPUT_BYTES * 3 // 5)

    with pytest.raises(InvalidInputError, match="too large"):
        await _validate(
            mocker, graph, {}, nodes_input_masks={"store-0": {"input": value}}
        )

    construct.assert_not_awaited()


# A graph input stored as {"<key>":null}: braces, quotes, colon and null, and
# the empty overrides and credentials stored beside it as {}.
_NULL_INPUT_OVERHEAD = 2 + 2 + 1 + 4 + 2 + 2


@pytest.mark.asyncio
async def test_a_large_input_key_at_the_input_budget_is_accepted(
    mocker: MockerFixture, construct: AsyncMock
):
    key = "k" * (utils.MAX_RUN_INPUT_BYTES - _NULL_INPUT_OVERHEAD)

    await _validate(mocker, _inputs_named_x(1), {key: None})

    construct.assert_awaited_once()
    counted = sum(utils._run_input_sizes(_inputs_named_x(1), {key: None}, {}, {}))
    stored = len(utils.json_dumps({key: None})) + len("{}") + len("{}")
    assert counted == stored == utils.MAX_RUN_INPUT_BYTES


@pytest.mark.asyncio
async def test_a_large_input_key_over_the_input_budget_is_refused(
    mocker: MockerFixture, construct: AsyncMock
):
    key = "k" * (utils.MAX_RUN_INPUT_BYTES - _NULL_INPUT_OVERHEAD + 1)

    with pytest.raises(InvalidInputError, match="too large"):
        await _validate(mocker, _inputs_named_x(1), {key: None})

    construct.assert_not_awaited()


def test_the_input_budget_counts_every_key_of_what_the_run_stores():
    graph_inputs = {"x": "value", "unused": None}
    nodes_input_masks = {"node-1": {"value": 1}, "node-2": {}}
    credentials = CredentialsMetaInput(
        id="cred-1", provider=ProviderName("github"), type="api_key"
    )
    graph_credentials_inputs = {"github": credentials}

    counted = sum(
        utils._run_input_sizes(
            _inputs_named_x(0),
            graph_inputs,
            nodes_input_masks,
            graph_credentials_inputs,
        )
    )

    assert counted == sum(
        len(utils.json_dumps(mapping).encode("utf-8"))
        for mapping in (
            graph_inputs,
            nodes_input_masks,
            {"github": credentials.model_dump(mode="json", exclude_none=True)},
        )
    )


def test_credentials_copied_into_nodes_get_a_short_title(mocker: MockerFixture):
    graph = mocker.MagicMock()
    field_info = CredentialsFieldInfo.model_validate(
        {"credentials_provider": ["github"], "credentials_types": ["api_key"]},
        by_alias=True,
    )
    node_fields = {(f"node-{i}", "credentials") for i in range(3)}
    graph.regular_credentials_inputs = {"github": (field_info, node_fields, True)}
    credentials = CredentialsMetaInput(
        id="cred-1",
        title="t" * (utils.MAX_CREDENTIALS_TITLE_LENGTH + 1),
        provider=ProviderName("github"),
        type="api_key",
    )

    result = utils.make_node_credentials_input_map(graph, {"github": credentials})

    assert len(result) == 3
    for node_input in result.values():
        copied = cast(dict, node_input["credentials"])
        assert copied["id"] == "cred-1"
        assert copied["title"] == "t" * utils.MAX_CREDENTIALS_TITLE_LENGTH
