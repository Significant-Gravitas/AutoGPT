import asyncio
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.blocks.data_manipulation import CreateListBlock
from backend.blocks.io import AgentInputBlock
from backend.data.graph import GraphModel, NodeModel
from backend.executor.scheduler import (
    GraphExecutionJobArgs,
    Scheduler,
    _build_trigger,
    _execute_graph,
)
from backend.executor.utils import (
    add_graph_execution,
    validate_and_construct_node_execution_input,
)
from backend.util.exceptions import GraphValidationError
from backend.util.json import nesting_depth


@pytest.fixture
def graph_source(mocker):
    graph = GraphModel(
        id="graph-1",
        version=1,
        is_active=True,
        name="Depth boundary",
        description="",
        user_id="user-1",
        created_at=datetime.now(UTC),
        nodes=[
            NodeModel(
                id="input-node",
                block_id=AgentInputBlock().id,
                graph_id="graph-1",
                graph_version=1,
                input_default={"name": "topic"},
            ),
            NodeModel(
                id="list-node",
                block_id=CreateListBlock().id,
                graph_id="graph-1",
                graph_version=1,
                input_default={"values": []},
            ),
        ],
    )
    source = AsyncMock()
    source.get_graph.return_value = graph
    source.resolve_default_tenancy.return_value = (None, None)
    mocker.patch("backend.executor.utils.graph_db", source)
    mocker.patch("backend.executor.utils.execution_db", source)
    mocker.patch(
        "backend.executor.utils.get_database_manager_async_client", return_value=source
    )
    return source


def nested(depth, kind="list"):
    value = "abyss"
    for _ in range(depth):
        value = [value] if kind == "list" else {"value": value}
    return value


@pytest.mark.parametrize("depth", [0, 100, 101, 126])
@pytest.mark.parametrize("kind", ["list", "dict"])
async def test_storable_graph_inputs_are_preserved(graph_source, depth, kind):
    value = nested(depth, kind)
    _, starting, _, _ = await validate_and_construct_node_execution_input(
        graph_id="graph-1", user_id="user-1", graph_inputs={"topic": value}
    )
    assert dict(starting)["input-node"]["value"] == value
    graph_source.validate_graph_execution_permissions.assert_awaited_once_with(
        user_id="user-1", graph_id="graph-1", graph_version=1, is_sub_graph=False
    )


@pytest.mark.parametrize("depth", [127, 300, 5000])
async def test_raw_input_is_rejected_before_graph_fetch(graph_source, depth):
    with pytest.raises(GraphValidationError, match="topic"):
        await validate_and_construct_node_execution_input(
            graph_id="graph-1",
            user_id="user-1",
            graph_inputs={"topic": nested(depth)},
        )
    graph_source.get_graph.assert_not_awaited()


async def test_literal_masks_account_for_both_storage_wrappers(graph_source):
    _, starting, masks, _ = await validate_and_construct_node_execution_input(
        graph_id="graph-1",
        user_id="user-1",
        graph_inputs={},
        nodes_input_masks={"list-node": {"values": nested(125)}},
    )
    assert nesting_depth(masks) == 127
    assert nesting_depth(dict(starting)["list-node"]["values"]) == 125

    with pytest.raises(GraphValidationError) as exc:
        await validate_and_construct_node_execution_input(
            graph_id="graph-1",
            user_id="user-1",
            graph_inputs={},
            nodes_input_masks={"list-node": {"values": nested(126), "max_size": 1}},
        )
    assert set(exc.value.node_errors) == {"list-node"}
    assert set(exc.value.node_errors["list-node"]) == {"values"}


async def test_all_deep_mask_fields_are_reported_before_graph_fetch(graph_source):
    with pytest.raises(GraphValidationError) as exc:
        await validate_and_construct_node_execution_input(
            graph_id="graph-1",
            user_id="user-1",
            graph_inputs={},
            nodes_input_masks={
                "list-node": {"values": nested(5000), "max_size": 1},
                "unused-node": {"payload": nested(126)},
            },
        )
    assert {key: set(errors) for key, errors in exc.value.node_errors.items()} == {
        "list-node": {"values"},
        "unused-node": {"payload"},
    }
    graph_source.get_graph.assert_not_awaited()


@pytest.mark.parametrize(
    "source,depth",
    [(source, depth) for source in ("mask", "default") for depth in (127, 128, 300)]
    + [("dynamic_pin", 127), ("dynamic_pin", 128)],
)
async def test_constructed_values_are_checked_after_coercion(
    graph_source, source, depth
):
    graph = graph_source.get_graph.return_value
    mask = {}
    if source == "dynamic_pin":
        graph.nodes[1].input_default = {"values_$_0": nested(depth - 1)}
    elif source == "default":
        graph.nodes[1].input_default = {"values": json.dumps(nested(depth))}
    else:
        mask = {"list-node": {"values": json.dumps(nested(depth))}}

    async def construct():
        return await validate_and_construct_node_execution_input(
            graph_id="graph-1",
            user_id="user-1",
            graph_inputs={},
            nodes_input_masks=mask,
        )

    if depth == 127:
        _, starting, _, _ = await construct()
        assert nesting_depth(dict(starting)["list-node"]["values"]) == depth
    else:
        with pytest.raises(GraphValidationError) as exc:
            await construct()
        assert set(exc.value.node_errors["list-node"]) == {"values"}


@pytest.mark.parametrize("source", ["raw", "mask", "encoded"])
async def test_rejected_run_never_reaches_persistence(graph_source, source):
    inputs = {"topic": nested(127)} if source == "raw" else {}
    masks = (
        {"list-node": {"values": nested(126)}}
        if source == "mask"
        else (
            {"list-node": {"values": json.dumps(nested(128))}}
            if source == "encoded"
            else {}
        )
    )
    with pytest.raises(GraphValidationError):
        await add_graph_execution(
            graph_id="graph-1",
            user_id="user-1",
            inputs=inputs,
            nodes_input_masks=masks,
        )
    graph_source.create_graph_execution.assert_not_awaited()


@pytest.fixture
def schedule(mocker):
    scheduler = Scheduler(register_system_tasks=False, throwaway_jobstores=True)
    scheduler.scheduler = MagicMock()
    job = MagicMock(id="schedule-1")
    job.name = "Daily"
    job.trigger = _build_trigger(cron="0 9 * * *", run_at=None, user_timezone="UTC")
    job.next_run_time = None
    job.kwargs = GraphExecutionJobArgs(
        schedule_id="schedule-1",
        user_id="user-1",
        graph_id="graph-1",
        graph_version=1,
        cron="0 9 * * *",
        input_data={"topic": "previous"},
    ).model_dump(mode="json")
    scheduler.scheduler.get_job.return_value = job
    scheduler.scheduler.modify_job.return_value = job
    mocker.patch.object(scheduler, "_persist_schedule", return_value=job)
    mocker.patch.object(scheduler, "_invalidate_jobs_cache")
    loop = asyncio.new_event_loop()
    mocker.patch(
        "backend.executor.scheduler.run_async", side_effect=loop.run_until_complete
    )
    mocker.patch("backend.executor.schedule_events.record_schedule_created")
    try:
        yield scheduler
    finally:
        loop.close()
        if scheduler._throwaway_dir is not None:
            scheduler._throwaway_dir.cleanup()


@pytest.mark.parametrize("operation", ["create", "update"])
@pytest.mark.parametrize("depth", [101, 126, 127])
def test_schedule_validates_before_persisting(graph_source, schedule, operation, depth):
    def save():
        if operation == "create":
            return schedule.add_graph_execution_schedule(
                graph_id="graph-1",
                graph_version=1,
                user_id="user-1",
                cron="0 9 * * *",
                input_data={"topic": nested(depth)},
                input_credentials={},
                user_timezone="UTC",
            )
        return schedule.update_graph_execution_schedule(
            schedule_id="schedule-1",
            user_id="user-1",
            input_data={"topic": nested(depth)},
        )

    if depth == 127:
        with pytest.raises(GraphValidationError, match="topic"):
            save()
        schedule._persist_schedule.assert_not_called()
        schedule.scheduler.modify_job.assert_not_called()
        assert schedule.scheduler.get_job.return_value.kwargs["input_data"] == {
            "topic": "previous"
        }
    else:
        save()
        if operation == "create":
            schedule._persist_schedule.assert_called_once()
        else:
            schedule.scheduler.modify_job.assert_called_once()


@pytest.mark.parametrize("operation", ["create", "update"])
def test_schedule_refuses_defaults_that_expand_past_storage(
    graph_source, schedule, operation
):
    graph_source.get_graph.return_value.nodes[1].input_default = {
        "values": json.dumps(nested(128))
    }
    with pytest.raises(GraphValidationError) as exc:
        if operation == "create":
            schedule.add_graph_execution_schedule(
                graph_id="graph-1",
                graph_version=1,
                user_id="user-1",
                cron="0 9 * * *",
                input_data={},
                input_credentials={},
                user_timezone="UTC",
            )
        else:
            schedule.update_graph_execution_schedule(
                schedule_id="schedule-1", user_id="user-1", input_data={}
            )
    assert set(exc.value.node_errors["list-node"]) == {"values"}
    schedule._persist_schedule.assert_not_called()
    schedule.scheduler.modify_job.assert_not_called()


@pytest.mark.parametrize("depth", [101, 126])
async def test_existing_storable_schedule_reaches_run_creation(
    graph_source, mocker, depth
):
    graph_source.create_graph_execution.side_effect = RuntimeError(
        "persistence boundary"
    )
    mocker.patch(
        "backend.executor.scheduler.get_database_manager_async_client",
        return_value=graph_source,
    )
    scheduler_client = AsyncMock()
    mocker.patch(
        "backend.executor.scheduler.get_scheduler_client", return_value=scheduler_client
    )
    await _execute_graph(
        schedule_id="schedule-1",
        user_id="user-1",
        graph_id="graph-1",
        graph_version=1,
        cron="0 9 * * *",
        input_data={"topic": nested(depth)},
    )
    graph_source.create_graph_execution.assert_awaited_once()
    scheduler_client.delete_schedule.assert_not_called()
