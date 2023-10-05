import io
import logging
from pathlib import Path
from random import randint
from typing import Any, Dict, List
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global artifacts store
artifacts: List[Dict[str, Any]] = []


class Task(BaseModel):
    input: str


class Artifact(BaseModel):
    binary: bytes
    relative_path: str
    file_name: str
    artifact_id: str


def generate_artifact_id() -> str:
    """
    Generate a unique artifact ID.
    """
    artifact_id = str(randint(0, 100000))
    while artifact_id in [artifact["artifact_id"] for artifact in artifacts]:
        artifact_id = str(randint(0, 100000))
    return artifact_id


def find_artifact_by_id(artifact_id: str) -> Dict[str, Any]:
    """
    Retrieve an artifact by its ID.
    """
    for artifact in artifacts:
        if artifact["artifact_id"] == artifact_id:
            return artifact
    return {}


@app.post("/agent/tasks/{task_id}/artifacts")
async def upload_file(
    task_id: str, file: Annotated[UploadFile, File()], relative_path: str = Form("")
) -> Dict[str, Any]:
    logger.info(
        "Uploading file for task_id: %s with relative path: %s", task_id, relative_path
    )
    absolute_directory_path = Path(__file__).parent.absolute()
    save_path = (
        absolute_directory_path
        / "agent/gpt-engineer"
        / "projects/my-new-project/workspace"
    )

    # Generate a unique artifact_id using UUID
    artifact_id = str(uuid.uuid4())

    artifact_data = await file.read()
    artifacts.append(
        {
            "binary": artifact_data,
            "relative_path": relative_path,
            "file_name": file.filename,
            "artifact_id": artifact_id,
        }
    )

    print(artifacts)
    return {
        "artifact_id": artifact_id,
        "file_name": "file_name",
        "relative_path": "relative_path",
    }

@app.get("/agent/tasks/{task_id}/artifacts")
async def get_files() -> List[Dict[str, Any]]:
    logger.info("Fetching list of files for task")
    return artifacts


@app.get("/agent/tasks/{task_id}/artifacts/{artifact_id}")
async def get_file(artifact_id: str):
    artifact = find_artifact_by_id(artifact_id)
    if not artifact:
        logger.error(f"Attempt to access nonexistent artifact with ID: {artifact_id}")
        raise HTTPException(status_code=404, detail="Artifact not found")

    logger.info(f"Fetching artifact with ID: {artifact_id}")
    return StreamingResponse(
        io.BytesIO(artifact["binary"]),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={artifact['file_name']}"},
    )


@app.post("/agent/tasks/{task_id}/steps")
async def create_steps(task_id: str):
    logger.info(f"Creating step for task_id: {task_id}")
    return {
        "input": "random",
        "additional_input": {},
        "task_id": task_id,
        "step_id": "random_step",
        "name": "random",
        "status": "created",
        "output": "random",
        "additional_output": {},
        "artifacts": [],
        "is_last": True,
    }


@app.post("/agent/tasks")
async def create_tasks(task: Task):
    artifacts.clear()
    return {
        "input": "random",
        "additional_input": {},
        "task_id": "static_task_id",
        "artifacts": [],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
