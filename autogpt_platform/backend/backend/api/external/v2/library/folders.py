"""V2 External API - Library Folder Endpoints"""

import logging
from typing import Optional

from fastapi import APIRouter, Query, Security
from prisma.enums import APIKeyPermission
from starlette import status

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

folders_router = APIRouter(tags=["library"])


@folders_router.get(
    path="/folders",
    summary="List folders in library",
    operation_id="listLibraryFolders",
)
async def list_folders(
    parent_id: Optional[str] = Query(
        default=None, description="Filter by parent folder ID. Omit for root folders."
    ),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_LIBRARY)
    ),
) -> LibraryFolderListResponse:
    """List folders in the user's library."""
    folders = await library_db.list_folders(
        user_id=auth.user_id,
        parent_id=parent_id,
    )

    return LibraryFolderListResponse(
        folders=[LibraryFolder.from_internal(f) for f in folders],
    )


@folders_router.get(
    path="/folders/tree",
    summary="Get library folder tree",
    operation_id="getLibraryFolderTree",
)
async def get_folder_tree(
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_LIBRARY)
    ),
) -> LibraryFolderTreeResponse:
    """Get the full folder tree for the user's library."""
    tree = await library_db.get_folder_tree(user_id=auth.user_id)

    return LibraryFolderTreeResponse(
        tree=[LibraryFolderTree.from_internal(f) for f in tree],
    )


@folders_router.get(
    path="/folders/{folder_id}",
    summary="Get folder in library",
    operation_id="getLibraryFolder",
)
async def get_folder(
    folder_id: str,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_LIBRARY)
    ),
) -> LibraryFolder:
    """Get details of a specific folder."""
    folder = await library_db.get_folder(
        folder_id=folder_id,
        user_id=auth.user_id,
    )
    return LibraryFolder.from_internal(folder)


@folders_router.post(
    path="/folders",
    summary="Create folder in library",
    operation_id="createLibraryFolder",
    status_code=status.HTTP_201_CREATED,
)
async def create_folder(
    request: LibraryFolderCreateRequest,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_LIBRARY)
    ),
) -> LibraryFolder:
    """Create a new folder in the user's library."""
    folder = await library_db.create_folder(
        user_id=auth.user_id,
        name=request.name,
        parent_id=request.parent_id,
        icon=request.icon,
        color=request.color,
    )
    return LibraryFolder.from_internal(folder)


@folders_router.patch(
    path="/folders/{folder_id}",
    summary="Update folder in library",
    operation_id="updateLibraryFolder",
)
async def update_folder(
    request: LibraryFolderUpdateRequest,
    folder_id: str,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_LIBRARY)
    ),
) -> LibraryFolder:
    """Update properties of a folder."""
    folder = await library_db.update_folder(
        folder_id=folder_id,
        user_id=auth.user_id,
        name=request.name,
        icon=request.icon,
        color=request.color,
    )
    return LibraryFolder.from_internal(folder)


@folders_router.post(
    path="/folders/{folder_id}/move",
    summary="Move folder in library",
    operation_id="moveLibraryFolder",
)
async def move_folder(
    request: LibraryFolderMoveRequest,
    folder_id: str,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_LIBRARY)
    ),
) -> LibraryFolder:
    """Move a folder to a new parent. Set target_parent_id to null to move to root."""
    folder = await library_db.move_folder(
        folder_id=folder_id,
        user_id=auth.user_id,
        target_parent_id=request.target_parent_id,
    )
    return LibraryFolder.from_internal(folder)


@folders_router.delete(
    path="/folders/{folder_id}",
    summary="Delete folder in library",
    operation_id="deleteLibraryFolder",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_folder(
    folder_id: str,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_LIBRARY)
    ),
) -> None:
    """
    Delete a folder and its subfolders. Agents in this folder will be moved to root.
    """
    await library_db.delete_folder(
        folder_id=folder_id,
        user_id=auth.user_id,
    )
