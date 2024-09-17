import typing

import fastapi

import autogpt_server.data.graph
import autogpt_server.data.execution
import autogpt_server.server.model
import autogpt_server.server.utils

router = fastapi.APIRouter()


@router.get("/graphs")
async def get_graphs(
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_server.server.utils.get_user_id)
    ]
) -> list[autogpt_server.data.graph.GraphMeta]:
    return await autogpt_server.data.graph.get_graphs_meta(
        filter_by="active", user_id=user_id
    )


@router.post("/graphs")
async def create_new_graph(
    create_graph: autogpt_server.server.model.CreateGraph,
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_server.server.utils.get_user_id)
    ],
) -> autogpt_server.data.graph.Graph:

    if create_graph.graph:
        graph = create_graph.graph
    elif create_graph.template_id:
        # Create a new graph from a template
        graph = await autogpt_server.data.graph.get_graph(
            create_graph.template_id,
            create_graph.template_version,
            template=True,
            user_id=user_id,
        )
        if not graph:
            raise fastapi.HTTPException(
                400, detail=f"Template #{create_graph.template_id} not found"
            )
        graph.version = 1
    else:
        raise fastapi.HTTPException(
            status_code=400, detail="Either graph or template_id must be provided."
        )

    graph.is_template = False
    graph.is_active = True
    graph.reassign_ids(reassign_graph_id=True)

    return await autogpt_server.data.graph.create_graph(graph, user_id=user_id)
    return await cls.create_graph(create_graph, is_template=False, user_id=user_id)


@router.get("/graphs/{graph_id}")
async def get_graph(
    graph_id: str,
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_server.server.utils.get_user_id)
    ],
    version: typing.Optional[int] = None,
) -> autogpt_server.data.graph.Graph:
    graph = await autogpt_server.data.graph.get_graph(
        graph_id, version, user_id=user_id
    )
    if not graph:
        raise fastapi.HTTPException(
            status_code=404, detail=f"Graph #{graph_id} not found."
        )
    return graph


@router.put("/graphs/{graph_id}")
async def update_graph(
    graph_id: str,
    graph: autogpt_server.data.graph.Graph,
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_server.server.utils.get_user_id)
    ],
) -> autogpt_server.data.graph.Graph:
    # Sanity check
    if graph.id and graph.id != graph_id:
        raise fastapi.HTTPException(400, detail="Graph ID does not match ID in URI")

    # Determine new version
    existing_versions = await autogpt_server.data.graph.get_graph_all_versions(
        graph_id, user_id=user_id
    )
    if not existing_versions:
        raise fastapi.HTTPException(404, detail=f"Graph #{graph_id} not found")
    latest_version_number = max(g.version for g in existing_versions)
    graph.version = latest_version_number + 1

    latest_version_graph = next(
        v for v in existing_versions if v.version == latest_version_number
    )
    if latest_version_graph.is_template != graph.is_template:
        raise fastapi.HTTPException(
            400, detail="Changing is_template on an existing graph is forbidden"
        )
    graph.is_active = not graph.is_template
    graph.reassign_ids()

    new_graph_version = await autogpt_server.data.graph.create_graph(
        graph, user_id=user_id
    )

    if new_graph_version.is_active:
        # Ensure new version is the only active version
        await autogpt_server.data.graph.set_graph_active_version(
            graph_id=graph_id, version=new_graph_version.version, user_id=user_id
        )

    return new_graph_version


@router.get("/graphs/{graph_id}/versions")
async def get_graph_all_versions(
    graph_id: str,
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_server.server.utils.get_user_id)
    ],
) -> list[autogpt_server.data.graph.Graph]:
    graphs = await autogpt_server.data.graph.get_graph_all_versions(
        graph_id, user_id=user_id
    )
    if not graphs:
        raise fastapi.HTTPException(
            status_code=404, detail=f"Graph #{graph_id} not found."
        )
    return graphs


@router.get("/graphs/{graph_id}/versions/{version}")
async def get_graph_version(
    graph_id: str,
    version: int,
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_server.server.utils.get_user_id)
    ],
) -> autogpt_server.data.graph.Graph:
    graph = await autogpt_server.data.graph.get_graph(
        graph_id, version, user_id=user_id
    )
    if not graph:
        raise fastapi.HTTPException(
            status_code=404, detail=f"Graph #{graph_id} not found."
        )
    return graph


@router.put("/graphs/{graph_id}/versions/active")
async def set_graph_active_version(
    graph_id: str,
    request_body: autogpt_server.server.model.SetGraphActiveVersion,
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_server.server.utils.get_user_id)
    ],
):
    new_active_version = request_body.active_graph_version
    if not await autogpt_server.data.graph.get_graph(
        graph_id, new_active_version, user_id=user_id
    ):
        raise fastapi.HTTPException(
            status_code=404, detail=f"Graph #{graph_id} v{new_active_version} not found"
        )
    await autogpt_server.data.graph.set_graph_active_version(
        graph_id=graph_id,
        version=request_body.active_graph_version,
        user_id=user_id,
    )


@router.post("/graphs/{graph_id}/execute")
async def execute_graph(
    graph_id: str,
    node_input: dict[typing.Any, typing.Any],
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_server.server.utils.get_user_id)
    ],
) -> dict[str, typing.Any]:  # FIXME: add proper return type
    try:
        graph_exec = self.execution_manager_client.add_execution(
            graph_id, node_input, user_id=user_id
        )
        return {"id": graph_exec["graph_exec_id"]}
    except Exception as e:
        msg = e.__str__().encode().decode("unicode_escape")
        raise fastapi.HTTPException(status_code=400, detail=msg)


@router.get("/graphs/{graph_id}/executions")
async def list_graph_runs(
    graph_id: str,
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_server.server.utils.get_user_id)
    ],
    graph_version: typing.Optional[int] = None,
) -> list[str]:
    graph = await autogpt_server.data.graph.get_graph(
        graph_id, graph_version, user_id=user_id
    )
    if not graph:
        rev = "" if graph_version is None else f" v{graph_version}"
        raise fastapi.HTTPException(
            status_code=404, detail=f"Agent #{graph_id}{rev} not found."
        )

    return await autogpt_server.data.execution.list_executions(graph_id, graph_version)


@router.get("/graphs/{graph_id}/executions/{graph_exec_id}")
async def get_graph_run_node_execution_results(
    graph_id: str,
    graph_exec_id: str,
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_server.server.utils.get_user_id)
    ],
) -> list[autogpt_server.data.execution.ExecutionResult]:
    graph = await autogpt_server.data.graph.get_graph(graph_id, user_id=user_id)
    if not graph:
        raise fastapi.HTTPException(
            status_code=404, detail=f"Graph #{graph_id} not found."
        )

    return await autogpt_server.data.execution.get_execution_results(graph_exec_id)


@router.post("/graphs/{graph_id}/executions/{graph_exec_id}/stop")
async def stop_graph_run(
    graph_exec_id: str,
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_server.server.utils.get_user_id)
    ],
) -> list[autogpt_server.data.execution.ExecutionResult]:
    if not await autogpt_server.data.execution.get_graph_execution(
        graph_exec_id, user_id
    ):
        raise fastapi.HTTPException(
            404, detail=f"Agent execution #{graph_exec_id} not found"
        )

    self.execution_manager_client.cancel_execution(graph_exec_id)

    # Retrieve & return canceled graph execution in its final state
    return await autogpt_server.data.execution.get_execution_results(graph_exec_id)


@router.post("/graphs/{graph_id}/schedules")
async def create_schedule(
    graph_id: str,
    cron: str,
    input_data: dict[typing.Any, typing.Any],
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_server.server.utils.get_user_id)
    ],
) -> dict[typing.Any, typing.Any]:
    graph = await autogpt_server.data.graph.get_graph(graph_id, user_id=user_id)
    if not graph:
        raise fastapi.HTTPException(
            status_code=404, detail=f"Graph #{graph_id} not found."
        )
    execution_scheduler = self.execution_scheduler_client
    return {
        "id": execution_scheduler.add_execution_schedule(
            graph_id, graph.version, cron, input_data, user_id=user_id
        )
    }


@router.get("/graphs/{graph_id}/schedules")
async def get_execution_schedules(
    graph_id: str,
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_server.server.utils.get_user_id)
    ],
) -> dict[str, str]:
    execution_scheduler = self.execution_scheduler_client
    return execution_scheduler.get_execution_schedules(graph_id, user_id)


@router.put("/graphs/schedules/{schedule_id}")
async def update_schedule(
    schedule_id: str,
    input_data: dict[typing.Any, typing.Any],
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_server.server.utils.get_user_id)
    ],
) -> dict[typing.Any, typing.Any]:
    execution_scheduler = self.execution_scheduler_client
    is_enabled = input_data.get("is_enabled", False)
    execution_scheduler.update_schedule(schedule_id, is_enabled, user_id=user_id)
    return {"id": schedule_id}
