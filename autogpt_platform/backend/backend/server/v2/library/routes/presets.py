import logging
import typing

import autogpt_libs.auth.depends
import autogpt_libs.auth.middleware
import autogpt_libs.utils.cache
import fastapi

import backend.data.graph
import backend.executor
import backend.integrations.creds_manager
import backend.integrations.webhooks.graph_lifecycle_hooks
import backend.server.v2.library.db
import backend.server.v2.library.model
import backend.util.service

logger = logging.getLogger(__name__)

router = fastapi.APIRouter()
integration_creds_manager = (
    backend.integrations.creds_manager.IntegrationCredentialsManager()
)


@autogpt_libs.utils.cache.thread_cached
def execution_manager_client() -> backend.executor.ExecutionManager:
    return backend.util.service.get_service_client(backend.executor.ExecutionManager)


@router.get("/presets")
async def get_presets(
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_libs.auth.depends.get_user_id)
    ],
    page: int = 1,
    page_size: int = 10,
) -> backend.server.v2.library.model.LibraryAgentPresetResponse:
    try:
        presets = await backend.server.v2.library.db.get_presets(
            user_id, page, page_size
        )
        return presets
    except Exception as e:
        logger.exception(f"Exception occurred whilst getting presets: {e}")
        raise fastapi.HTTPException(status_code=500, detail="Failed to get presets")


@router.get("/presets/{preset_id}")
async def get_preset(
    preset_id: str,
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_libs.auth.depends.get_user_id)
    ],
) -> backend.server.v2.library.model.LibraryAgentPreset:
    try:
        preset = await backend.server.v2.library.db.get_preset(user_id, preset_id)
        if not preset:
            raise fastapi.HTTPException(
                status_code=404,
                detail=f"Preset {preset_id} not found",
            )
        return preset
    except Exception as e:
        logger.exception(f"Exception occurred whilst getting preset: {e}")
        raise fastapi.HTTPException(status_code=500, detail="Failed to get preset")


@router.post("/presets")
async def create_preset(
    preset: backend.server.v2.library.model.CreateLibraryAgentPresetRequest,
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_libs.auth.depends.get_user_id)
    ],
) -> backend.server.v2.library.model.LibraryAgentPreset:
    try:
        return await backend.server.v2.library.db.create_or_update_preset(
            user_id, preset
        )
    except Exception as e:
        logger.exception(f"Exception occurred whilst creating preset: {e}")
        raise fastapi.HTTPException(status_code=500, detail="Failed to create preset")


@router.put("/presets/{preset_id}")
async def update_preset(
    preset_id: str,
    preset: backend.server.v2.library.model.CreateLibraryAgentPresetRequest,
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_libs.auth.depends.get_user_id)
    ],
) -> backend.server.v2.library.model.LibraryAgentPreset:
    try:
        return await backend.server.v2.library.db.create_or_update_preset(
            user_id, preset, preset_id
        )
    except Exception as e:
        logger.exception(f"Exception occurred whilst updating preset: {e}")
        raise fastapi.HTTPException(status_code=500, detail="Failed to update preset")


@router.delete("/presets/{preset_id}")
async def delete_preset(
    preset_id: str,
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_libs.auth.depends.get_user_id)
    ],
):
    try:
        await backend.server.v2.library.db.delete_preset(user_id, preset_id)
        return fastapi.Response(status_code=204)
    except Exception as e:
        logger.exception(f"Exception occurred whilst deleting preset: {e}")
        raise fastapi.HTTPException(status_code=500, detail="Failed to delete preset")


@router.post(
    path="/presets/{preset_id}/execute",
    tags=["presets"],
    dependencies=[fastapi.Depends(autogpt_libs.auth.middleware.auth_middleware)],
)
async def execute_preset(
    graph_id: str,
    graph_version: int,
    preset_id: str,
    node_input: dict[typing.Any, typing.Any],
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_libs.auth.depends.get_user_id)
    ],
) -> dict[str, typing.Any]:  # FIXME: add proper return type
    try:
        preset = await backend.server.v2.library.db.get_preset(user_id, preset_id)
        if not preset:
            raise fastapi.HTTPException(status_code=404, detail="Preset not found")

        logger.info(f"Preset inputs: {preset.inputs}")

        updated_node_input = node_input.copy()
        # Merge in preset input values
        for key, value in preset.inputs.items():
            if key not in updated_node_input:
                updated_node_input[key] = value

        execution = execution_manager_client().add_execution(
            graph_id=graph_id,
            graph_version=graph_version,
            data=updated_node_input,
            user_id=user_id,
            preset_id=preset_id,
        )

        logger.info(f"Execution added: {execution} with input: {updated_node_input}")

        return {"id": execution.graph_exec_id}
    except Exception as e:
        msg = e.__str__().encode().decode("unicode_escape")
        raise fastapi.HTTPException(status_code=400, detail=msg)
