import io
import json
import logging
import shutil
from pathlib import Path
from random import randint
from typing import Annotated, Any, Dict, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
artifacts: List[Dict[str, Any]] = []


class Task(BaseModel):
    input: str


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

    random_string = str(randint(0, 100000))
    while random_string in artifacts:
        random_string = str(randint(0, 100000))

    artifact_data = await file.read()
    artifacts.append(
        {
            "binary": artifact_data,
            "relative_path": relative_path,
            "file_name": file.filename,
            "artifact_id": random_string,
        }
    )

    print(artifacts)
    return {
        "artifact_id": random_string,
        "file_name": "file_name",
        "relative_path": "relative_path",
    }


@app.get("/agent/tasks/{task_id}/artifacts")
async def get_files() -> List[Dict[str, Any]]:
    logger.info("Fetching list of files for task")
    return artifacts


@app.get("/agent/tasks/{task_id}/artifacts/{artifact_id}")
async def get_file(artifact_id: str):
    for artifact in artifacts:
        if artifact["artifact_id"] == artifact_id:
            break
    else:
        logger.error("Attempt to access nonexistent artifact with ID: %s", artifact_id)
        raise HTTPException(status_code=404, detail="Artifact not found")

    logger.info("Fetching artifact with ID: %s", artifact_id)
    # find aritifact where artifact_id = artifact_id

    for artifact in artifacts:
        if artifact["artifact_id"] == artifact_id:
            return StreamingResponse(
                io.BytesIO(artifact["binary"]),
                media_type="application/octet-stream",
                headers={"Content-Disposition": f"attachment; filename=test.txt"},
            )
    # return 404
    return HTTPException(status_code=404, detail="Artifact not found")


@app.post("/agent/tasks/{task_id}/steps")
async def create_steps(task_id: str):
    logger.info("Creating step for task_id: %s", task_id)
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
