from typing import cast
from unittest.mock import AsyncMock

import pytest
from pytest_mock import MockerFixture

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
    # Eight copies of a JSON string whose quotes make it exactly an eighth.
    value = "a" * (utils.MAX_RUN_INPUT_BYTES // 8 - 2)

    await _validate(mocker, _inputs_named_x(8), {"x": value})

    construct.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_run_over_the_input_budget_is_refused_before_it_is_built(
    mocker: MockerFixture, construct: AsyncMock
):
    measure = mocker.spy(utils, "json_dumps")
    value = "a" * (utils.MAX_RUN_INPUT_BYTES // 8 - 1)

    with pytest.raises(InvalidInputError, match="too large"):
        await _validate(mocker, _inputs_named_x(8), {"x": value})

    construct.assert_not_awaited()
    assert measure.call_count == 1


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
        title="t" * 100_000,
        provider=ProviderName("github"),
        type="api_key",
    )

    result = utils.make_node_credentials_input_map(graph, {"github": credentials})

    assert len(result) == 3
    for node_input in result.values():
        copied = cast(dict, node_input["credentials"])
        assert copied["id"] == "cred-1"
        assert copied["title"] == "t" * utils.MAX_CREDENTIALS_TITLE_LENGTH
