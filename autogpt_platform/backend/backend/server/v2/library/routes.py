import logging
import typing

import autogpt_libs.auth.depends
import autogpt_libs.auth.middleware
import fastapi

import backend.data.graph
import backend.server.v2.library.db
import backend.server.v2.library.model

logger = logging.getLogger(__name__)

router = fastapi.APIRouter()


@router.get(
    "/agents",
    tags=["library", "private"],
    dependencies=[fastapi.Depends(autogpt_libs.auth.middleware.auth_middleware)],
)
async def get_library_agents(
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_libs.auth.depends.get_user_id)
    ]
) -> typing.Sequence[backend.server.v2.library.model.LibraryAgent]:
    """
    Get all agents in the user's library, including both created and saved agents.
    """
    try:
        agents = await backend.server.v2.library.db.get_library_agents(user_id)
        return agents
    except Exception:
        logger.exception("Exception occurred whilst getting library agents")
        raise fastapi.HTTPException(
            status_code=500, detail="Failed to get library agents"
        )


@router.post(
    "/agents/{store_listing_version_id}",
    tags=["library", "private"],
    dependencies=[fastapi.Depends(autogpt_libs.auth.middleware.auth_middleware)],
    status_code=201,
)
async def add_agent_to_library(
    store_listing_version_id: str,
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_libs.auth.depends.get_user_id)
    ],
) -> fastapi.Response:
    """
    Add an agent from the store to the user's library.

    Args:
        store_listing_version_id (str): ID of the store listing version to add
        user_id (str): ID of the authenticated user

    Returns:
        fastapi.Response: 201 status code on success

    Raises:
        HTTPException: If there is an error adding the agent to the library
    """
    try:
        await backend.server.v2.library.db.add_agent_to_library(
            store_listing_version_id=store_listing_version_id, user_id=user_id
        )
        return fastapi.Response(status_code=201)
    except Exception:
        logger.exception("Exception occurred whilst adding agent to library")
        raise fastapi.HTTPException(
            status_code=500, detail="Failed to add agent to library"
        )
