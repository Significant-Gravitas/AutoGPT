import logging
from typing import Optional

import autogpt_libs.auth as autogpt_auth_lib
from fastapi import APIRouter, HTTPException, Query, Security, status
from fastapi.responses import Response

from backend.util.exceptions import DatabaseError, NotFoundError

from .. import db as library_db
from .. import model as library_model

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/folders",
    tags=["library", "folders", "private"],
    dependencies=[Security(autogpt_auth_lib.requires_user)],
)


@router.get(
    "",
    summary="List Library Folders",
    response_model=library_model.FolderListResponse,
    responses={
        200: {"description": "List of folders"},
        500: {"description": "Server error"},
    },
)
async def list_folders(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
    parent_id: Optional[str] = Query(
        None,
        description="Filter by parent folder ID. If not provided, returns root-level folders.",
    ),
    include_counts: bool = Query(
        True,
        description="Include agent and subfolder counts",
    ),
) -> library_model.FolderListResponse:
    """
    List folders for the authenticated user.

    Args:
        user_id: ID of the authenticated user.
        parent_id: Optional parent folder ID to filter by.
        include_counts: Whether to include agent and subfolder counts.

    Returns:
        A FolderListResponse containing folders.
    """
    try:
        folders = await library_db.list_folders(
            user_id=user_id,
            parent_id=parent_id,
            include_counts=include_counts,
        )
        return library_model.FolderListResponse(
            folders=folders,
            pagination=library_model.Pagination(
                total_items=len(folders),
                total_pages=1,
                current_page=1,
                page_size=len(folders),
            ),
        )
    except Exception as e:
        logger.error(f"Could not list folders for user #{user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from e


@router.get(
    "/tree",
    summary="Get Folder Tree",
    response_model=library_model.FolderTreeResponse,
    responses={
        200: {"description": "Folder tree structure"},
        500: {"description": "Server error"},
    },
)
async def get_folder_tree(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> library_model.FolderTreeResponse:
    """
    Get the full folder tree for the authenticated user.

    Args:
        user_id: ID of the authenticated user.

    Returns:
        A FolderTreeResponse containing the nested folder structure.
    """
    try:
        tree = await library_db.get_folder_tree(user_id=user_id)
        return library_model.FolderTreeResponse(tree=tree)
    except Exception as e:
        logger.error(f"Could not get folder tree for user #{user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from e


@router.get(
    "/{folder_id}",
    summary="Get Folder",
    response_model=library_model.LibraryFolder,
    responses={
        200: {"description": "Folder details"},
        404: {"description": "Folder not found"},
        500: {"description": "Server error"},
    },
)
async def get_folder(
    folder_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> library_model.LibraryFolder:
    """
    Get a specific folder.

    Args:
        folder_id: ID of the folder to retrieve.
        user_id: ID of the authenticated user.

    Returns:
        The requested LibraryFolder.
    """
    try:
        return await library_db.get_folder(folder_id=folder_id, user_id=user_id)
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Could not get folder #{folder_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from e


@router.post(
    "",
    summary="Create Folder",
    status_code=status.HTTP_201_CREATED,
    response_model=library_model.LibraryFolder,
    responses={
        201: {"description": "Folder created successfully"},
        400: {"description": "Validation error"},
        404: {"description": "Parent folder not found"},
        409: {"description": "Folder name conflict"},
        500: {"description": "Server error"},
    },
)
async def create_folder(
    payload: library_model.FolderCreateRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> library_model.LibraryFolder:
    """
    Create a new folder.

    Args:
        payload: The folder creation request.
        user_id: ID of the authenticated user.

    Returns:
        The created LibraryFolder.
    """
    try:
        return await library_db.create_folder(
            user_id=user_id,
            name=payload.name,
            parent_id=payload.parent_id,
            icon=payload.icon,
            color=payload.color,
        )
    except library_db.FolderValidationError as e:
        if "already exists" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e),
            ) from e
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except DatabaseError as e:
        logger.error(f"Database error creating folder: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from e


@router.patch(
    "/{folder_id}",
    summary="Update Folder",
    response_model=library_model.LibraryFolder,
    responses={
        200: {"description": "Folder updated successfully"},
        400: {"description": "Validation error"},
        404: {"description": "Folder not found"},
        409: {"description": "Folder name conflict"},
        500: {"description": "Server error"},
    },
)
async def update_folder(
    folder_id: str,
    payload: library_model.FolderUpdateRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> library_model.LibraryFolder:
    """
    Update a folder's properties.

    Args:
        folder_id: ID of the folder to update.
        payload: The folder update request.
        user_id: ID of the authenticated user.

    Returns:
        The updated LibraryFolder.
    """
    try:
        return await library_db.update_folder(
            folder_id=folder_id,
            user_id=user_id,
            name=payload.name,
            icon=payload.icon,
            color=payload.color,
        )
    except library_db.FolderValidationError as e:
        if "already exists" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e),
            ) from e
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except DatabaseError as e:
        logger.error(f"Database error updating folder #{folder_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from e


@router.post(
    "/{folder_id}/move",
    summary="Move Folder",
    response_model=library_model.LibraryFolder,
    responses={
        200: {"description": "Folder moved successfully"},
        400: {"description": "Validation error (circular reference, depth exceeded)"},
        404: {"description": "Folder or target parent not found"},
        409: {"description": "Folder name conflict in target location"},
        500: {"description": "Server error"},
    },
)
async def move_folder(
    folder_id: str,
    payload: library_model.FolderMoveRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> library_model.LibraryFolder:
    """
    Move a folder to a new parent.

    Args:
        folder_id: ID of the folder to move.
        payload: The move request with target parent.
        user_id: ID of the authenticated user.

    Returns:
        The moved LibraryFolder.
    """
    try:
        return await library_db.move_folder(
            folder_id=folder_id,
            user_id=user_id,
            target_parent_id=payload.target_parent_id,
        )
    except library_db.FolderValidationError as e:
        if "already exists" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e),
            ) from e
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except DatabaseError as e:
        logger.error(f"Database error moving folder #{folder_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from e


@router.delete(
    "/{folder_id}",
    summary="Delete Folder",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        204: {"description": "Folder deleted successfully"},
        404: {"description": "Folder not found"},
        500: {"description": "Server error"},
    },
)
async def delete_folder(
    folder_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> Response:
    """
    Soft-delete a folder and all its contents.

    Args:
        folder_id: ID of the folder to delete.
        user_id: ID of the authenticated user.

    Returns:
        204 No Content if successful.
    """
    try:
        await library_db.delete_folder(
            folder_id=folder_id,
            user_id=user_id,
            soft_delete=True,
        )
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except DatabaseError as e:
        logger.error(f"Database error deleting folder #{folder_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from e


# === Bulk Agent Operations ===


@router.post(
    "/agents/bulk-move",
    summary="Bulk Move Agents",
    response_model=list[library_model.LibraryAgent],
    responses={
        200: {"description": "Agents moved successfully"},
        404: {"description": "Folder not found"},
        500: {"description": "Server error"},
    },
)
async def bulk_move_agents(
    payload: library_model.BulkMoveAgentsRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> list[library_model.LibraryAgent]:
    """
    Move multiple agents to a folder.

    Args:
        payload: The bulk move request with agent IDs and target folder.
        user_id: ID of the authenticated user.

    Returns:
        The updated LibraryAgents.
    """
    try:
        return await library_db.bulk_move_agents_to_folder(
            agent_ids=payload.agent_ids,
            folder_id=payload.folder_id,
            user_id=user_id,
        )
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except DatabaseError as e:
        logger.error(f"Database error bulk moving agents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from e
