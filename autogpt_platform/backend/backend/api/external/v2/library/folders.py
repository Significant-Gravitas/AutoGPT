"""
V2 External API - Library Folder Endpoints

Provides endpoints for managing library folders.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Path, Query, Security
from prisma.enums import APIKeyPermission
from starlette.status import HTTP_201_CREATED, HTTP_204_NO_CONTENT

from backend.api.external.middleware import require_permission
from backend.api.features.library import db as library_db
from backend.data.auth.base import APIAuthorizationInfo

from ..models import (
    LibraryFolder,
    LibraryFolderCreateRequest,
    LibraryFolderListResponse,
    LibraryFolderMoveRequest,
    LibraryFolderTree,
    LibraryFolderTreeResponse,
    LibraryFolderUpdateRequest,
)

logger = logging.getLogger(__name__)

folders_router = APIRouter()


@folders_router.get(
    path="/folders",
    summary="List folders",
)
async def list_folders(
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_LIBRARY)
    ),
    parent_id: Optional[str] = Query(
        default=None, description="Filter by parent folder ID (null = root folders)"
    ),
) -> LibraryFolderListResponse:
    """
    List folders in the user's library.

    Optionally filter by parent folder ID to list children of a specific folder.
    """
    folders = await library_db.list_folders(
        user_id=auth.user_id,
        parent_id=parent_id,
    )

    return LibraryFolderListResponse(
        folders=[LibraryFolder.from_internal(f) for f in folders],
    )


@folders_router.get(
    path="/folders/tree",
    summary="Get folder tree",
)
async def get_folder_tree(
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_LIBRARY)
    ),
) -> LibraryFolderTreeResponse:
    """
    Get the full folder tree for the user's library.

    Returns a hierarchical tree structure with nested children.
    """
    tree = await library_db.get_folder_tree(user_id=auth.user_id)

    return LibraryFolderTreeResponse(
        tree=[LibraryFolderTree.from_internal(f) for f in tree],
    )


@folders_router.get(
    path="/folders/{folder_id}",
    summary="Get folder",
)
async def get_folder(
    folder_id: str = Path(description="Folder ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_LIBRARY)
    ),
) -> LibraryFolder:
    """
    Get details of a specific folder.
    """
    try:
        folder = await library_db.get_folder(
            folder_id=folder_id,
            user_id=auth.user_id,
        )
    except Exception:
        raise HTTPException(status_code=404, detail=f"Folder #{folder_id} not found")

    return LibraryFolder.from_internal(folder)


@folders_router.post(
    path="/folders",
    summary="Create a folder",
    status_code=HTTP_201_CREATED,
)
async def create_folder(
    request: LibraryFolderCreateRequest,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_LIBRARY)
    ),
) -> LibraryFolder:
    """
    Create a new folder in the user's library.
    """
    try:
        folder = await library_db.create_folder(
            user_id=auth.user_id,
            name=request.name,
            parent_id=request.parent_id,
            icon=request.icon,
            color=request.color,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return LibraryFolder.from_internal(folder)


@folders_router.patch(
    path="/folders/{folder_id}",
    summary="Update a folder",
)
async def update_folder(
    request: LibraryFolderUpdateRequest,
    folder_id: str = Path(description="Folder ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_LIBRARY)
    ),
) -> LibraryFolder:
    """
    Update properties of a folder.

    Only the fields provided in the request body will be updated.
    """
    try:
        folder = await library_db.update_folder(
            folder_id=folder_id,
            user_id=auth.user_id,
            name=request.name,
            icon=request.icon,
            color=request.color,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return LibraryFolder.from_internal(folder)


@folders_router.post(
    path="/folders/{folder_id}/move",
    summary="Move a folder",
)
async def move_folder(
    request: LibraryFolderMoveRequest,
    folder_id: str = Path(description="Folder ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_LIBRARY)
    ),
) -> LibraryFolder:
    """
    Move a folder to a new parent.

    Set target_parent_id to null to move to root.
    """
    try:
        folder = await library_db.move_folder(
            folder_id=folder_id,
            user_id=auth.user_id,
            target_parent_id=request.target_parent_id,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return LibraryFolder.from_internal(folder)


@folders_router.delete(
    path="/folders/{folder_id}",
    summary="Delete a folder",
    status_code=HTTP_204_NO_CONTENT,
)
async def delete_folder(
    folder_id: str = Path(description="Folder ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_LIBRARY)
    ),
) -> None:
    """
    Delete a folder.

    Agents in the folder will be moved to root. Subfolders are also deleted.
    """
    try:
        await library_db.delete_folder(
            folder_id=folder_id,
            user_id=auth.user_id,
        )
    except Exception:
        raise HTTPException(status_code=404, detail=f"Folder #{folder_id} not found")
