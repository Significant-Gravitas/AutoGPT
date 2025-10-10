import logging
import time
from pathlib import Path
from typing import AsyncIterator, Optional, Dict, Any

from agent_protocol_client import (
    AgentApi,
    ApiClient,
    Configuration,
    Step,
    TaskRequestBody,
)

from agbenchmark.agent_interface import get_list_of_file_paths
from agbenchmark.config import AgentBenchmarkConfig

logger = logging.getLogger(__name__)


async def run_api_agent(
    task: str,
    config: AgentBenchmarkConfig,
    timeout: int,
    artifacts_location: Optional[Path] = None,
    *,
    mock: bool = False,
) -> AsyncIterator[Step]:
    """
    Run an agent API task, stream its steps, and track progress metrics.
    Yields each Step as it is produced.
    """
    configuration = Configuration(host=config.host)
    async with ApiClient(configuration) as api_client:
        api_instance = AgentApi(api_client)
        task_request_body = TaskRequestBody(input=task, additional_input=None)

        start_time = time.time()
        response = await api_instance.create_agent_task(task_request_body=task_request_body)
        task_id = response.task_id

        logger.info(f"Task created with ID: {task_id}")
        metrics: Dict[str, Any] = {
            "task_id": task_id,
            "steps_executed": 0,
            "start_time": start_time,
            "end_time": None,
            "status": "running",
        }

        # Upload input artifacts if any
        if artifacts_location:
            logger.debug("Uploading task input artifacts to agent...")
            await upload_artifacts(api_instance, artifacts_location, task_id, "artifacts_in")

        logger.debug("Running agent until finished or timeout...")
        while True:
            step = await api_instance.execute_agent_task_step(task_id=task_id)
            metrics["steps_executed"] += 1
            elapsed = time.time() - start_time

            logger.info(
                f"[Step {metrics['steps_executed']}] "
                f"Elapsed: {elapsed:.1f}s | Step ID: {getattr(step, 'step_id', 'N/A')}"
            )

            yield step  # Stream each step to the caller

            if elapsed > timeout:
                metrics["status"] = "timeout"
                raise TimeoutError("Time limit exceeded")

            if step and mock:
                step.is_last = True

            if not step or step.is_last:
                break

        # After completion
        metrics["end_time"] = time.time()
        metrics["duration"] = metrics["end_time"] - metrics["start_time"]
        metrics["status"] = "success"

        if artifacts_location:
            if mock:
                logger.debug("Uploading mock artifacts to agent...")
                await upload_artifacts(api_instance, artifacts_location, task_id, "artifacts_out")

            logger.debug("Downloading agent artifacts...")
            await download_agent_artifacts_into_folder(api_instance, task_id, config.temp_folder)

        logger.info(
            f"Task {task_id} finished. Steps: {metrics['steps_executed']} | "
            f"Duration: {metrics['duration']:.1f}s | Status: {metrics['status']}"
        )

        # Optionally, you could write metrics to a file or return them
        # For now, just log them
        logger.debug(f"Final Metrics: {metrics}")


async def download_agent_artifacts_into_folder(
    api_instance: AgentApi, task_id: str, folder: Path
):
    artifacts = await api_instance.list_agent_task_artifacts(task_id=task_id)

    for artifact in artifacts.artifacts:
        # Determine correct folder
        if artifact.relative_path:
            path: str = (
                artifact.relative_path
                if not artifact.relative_path.startswith("/")
                else artifact.relative_path[1:]
            )
            folder = (folder / path).parent

        if not folder.exists():
            folder.mkdir(parents=True)

        file_path = folder / artifact.file_name
        logger.debug(f"Downloading agent artifact {artifact.file_name} to {folder}")
        with open(file_path, "wb") as f:
            content = await api_instance.download_agent_task_artifact(
                task_id=task_id, artifact_id=artifact.artifact_id
            )
            f.write(content)


async def upload_artifacts(
    api_instance: AgentApi, artifacts_location: Path, task_id: str, type: str
) -> None:
    for file_path in get_list_of_file_paths(artifacts_location, type):
        relative_path: Optional[str] = "/".join(
            str(file_path).split(f"{type}/", 1)[-1].split("/")[:-1]
        )
        if not relative_path:
            relative_path = None

        logger.debug(f"Uploading artifact {file_path.name} (type={type})...")
        await api_instance.upload_agent_task_artifacts(
            task_id=task_id, file=str(file_path), relative_path=relative_path
        )
