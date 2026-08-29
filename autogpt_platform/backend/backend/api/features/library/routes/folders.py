from typing import Optional

import autogpt_libs.auth as autogpt_auth_lib
from fastapi import APIRouter, Query, Security, status
from fastapi.responses import Response

from .. import db as library_db
from .. import model as library_model

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
    include_relations: bool = Query(
        True,
        description="Include agent and subfolder relations (for counts)",
    ),
) -> library_model.FolderListResponse:
    """
    List folders for the authenticated user.

    Args:
        user_id: ID of the authenticated user.
        parent_id: Optional parent folder ID to filter by.
        include_relations: Whether to include agent and subfolder relations for counts.

    Returns:
        A FolderListResponse containing folders.
    """
    folders = await library_db.list_folders(
        user_id=user_id,
        parent_id=parent_id,
        include_relations=include_relations,
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
    tree = await library_db.get_folder_tree(user_id=user_id)
    return library_model.FolderTreeResponse(tree=tree)


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
    return await library_db.get_folder(folder_id=folder_id, user_id=user_id)


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
    return await library_db.create_folder(
        user_id=user_id,
        name=payload.name,
        parent_id=payload.parent_id,
        icon=payload.icon,
        color=payload.color,
    )


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
    return await library_db.update_folder(
        folder_id=folder_id,
        user_id=user_id,
        name=payload.name,
        icon=payload.icon,
        color=payload.color,
    )


@router.post(
    "/{folder_id}/move",
    summary="Move Folder",
    response_model=library_model.LibraryFolder,
    responses={
        200: {"description": "Folder moved successfully"},
        400: {"description": "Validation error (circular reference)"},
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
    return await library_db.move_folder(
        folder_id=folder_id,
        user_id=user_id,
        target_parent_id=payload.target_parent_id,
    )


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
    await library_db.delete_folder(
        folder_id=folder_id,
        user_id=user_id,
        soft_delete=True,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


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
    return await library_db.bulk_move_agents_to_folder(
        agent_ids=payload.agent_ids,
        folder_id=payload.folder_id,
        user_id=user_id,
    )
