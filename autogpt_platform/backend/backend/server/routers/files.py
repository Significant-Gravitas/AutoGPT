from typing import Annotated, Sequence

import fastapi

import backend.data.files as files_db
from backend.server.utils import get_user_id

files_api = fastapi.APIRouter()


@files_api.get(path="/", tags=["files"])
async def list_files(
    user_id: Annotated[str, fastapi.Depends(get_user_id)],
) -> Sequence[files_db.File]:
    return await files_db.list_files(user_id=user_id)


@files_api.post(path="/", tags=["files"])
async def upload_file(
    user_id: Annotated[str, fastapi.Depends(get_user_id)],
    file: fastapi.UploadFile,
) -> files_db.File:
    return await files_db.create_file_from_upload(user_id=user_id, uploaded_file=file)


@files_api.get(path="/{file_id}", tags=["files"])
async def get_file_meta(
    user_id: Annotated[str, fastapi.Depends(get_user_id)],
    file_id: Annotated[str, fastapi.Path()],
) -> files_db.File:
    return await files_db.get_file(user_id=user_id, file_id=file_id)


@files_api.get(path="/{file_id}/download", tags=["files"])
async def download_file(
    user_id: Annotated[str, fastapi.Depends(get_user_id)],
    file_id: Annotated[str, fastapi.Path()],
):
    file, blob = await files_db.get_file_content(user_id=user_id, file_id=file_id)
    return fastapi.responses.StreamingResponse(
        content=blob.open(),
        media_type=file.content_type,
        headers={"Content-Disposition": f'attachment; filename="{file.name}"'},
    )
