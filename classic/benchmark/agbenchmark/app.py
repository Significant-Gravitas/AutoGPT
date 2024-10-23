import datetime
import glob
import json
import logging
import sys
import time
import uuid
from collections import deque
from multiprocessing import Process
from pathlib import Path
from typing import Optional

import httpx
import psutil
from agent_protocol_client import AgentApi, ApiClient, ApiException, Configuration
from agent_protocol_client.models import Task, TaskRequestBody
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, ValidationError

from agbenchmark.challenges import ChallengeInfo
from agbenchmark.config import AgentBenchmarkConfig
from agbenchmark.reports.processing.report_types_v2 import (
    BenchmarkRun,
    Metrics,
    RepositoryInfo,
    RunDetails,
    TaskInfo,
)
from agbenchmark.schema import TaskEvalRequestBody
from agbenchmark.utils.utils import write_pretty_json

sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

CHALLENGES: dict[str, ChallengeInfo] = {}
challenges_path = Path(__file__).parent / "challenges"
challenge_spec_files = deque(
    glob.glob(
        f"{challenges_path}/**/data.json",
        recursive=True,
    )
)

logger.debug("Loading challenges...")
while challenge_spec_files:
    challenge_spec_file = Path(challenge_spec_files.popleft())
    challenge_relpath = challenge_spec_file.relative_to(challenges_path.parent)
    if challenge_relpath.is_relative_to("challenges/deprecated"):
        continue

    logger.debug(f"Loading {challenge_relpath}...")
    try:
        challenge_info = ChallengeInfo.model_validate_json(
            challenge_spec_file.read_text()
        )
    except ValidationError as e:
        if logging.getLogger().level == logging.DEBUG:
            logger.warning(f"Spec file {challenge_relpath} failed to load:\n{e}")
        logger.debug(f"Invalid challenge spec: {challenge_spec_file.read_text()}")
        continue

    if not challenge_info.eval_id:
        challenge_info.eval_id = str(uuid.uuid4())
        # this will sort all the keys of the JSON systematically
        # so that the order is always the same
        write_pretty_json(challenge_info.model_dump(), challenge_spec_file)

    CHALLENGES[challenge_info.eval_id] = challenge_info


class BenchmarkTaskInfo(BaseModel):
    task_id: str
    start_time: datetime.datetime
    challenge_info: ChallengeInfo


task_informations: dict[str, BenchmarkTaskInfo] = {}


def find_agbenchmark_without_uvicorn():
    pids = []
    for process in psutil.process_iter(
        attrs=[
            "pid",
            "cmdline",
            "name",
            "username",
            "status",
            "cpu_percent",
            "memory_info",
            "create_time",
            "cwd",
            "connections",
        ]
    ):
        try:
            # Convert the process.info dictionary values to strings and concatenate them
            full_info = " ".join([str(v) for k, v in process.as_dict().items()])

            if "agbenchmark" in full_info and "uvicorn" not in full_info:
                pids.append(process.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return pids


class CreateReportRequest(BaseModel):
    test: str
    test_run_id: str
    # category: Optional[str] = []
    mock: Optional[bool] = False

    model_config = ConfigDict(extra="forbid")


updates_list = []

origins = [
    "http://localhost:8000",
    "http://localhost:8080",
    "http://127.0.0.1:5000",
    "http://localhost:5000",
]


def stream_output(pipe):
    for line in pipe:
        print(line, end="")


def setup_fastapi_app(agbenchmark_config: AgentBenchmarkConfig) -> FastAPI:
    from agbenchmark.agent_api_interface import upload_artifacts
    from agbenchmark.challenges import get_challenge_from_source_uri
    from agbenchmark.main import run_benchmark

    configuration = Configuration(
        host=agbenchmark_config.host or "http://localhost:8000"
    )
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    router = APIRouter()

    @router.post("/reports")
    def run_single_test(body: CreateReportRequest) -> dict:
        pids = find_agbenchmark_without_uvicorn()
        logger.info(f"pids already running with agbenchmark: {pids}")

        logger.debug(f"Request to /reports: {body.model_dump()}")

        # Start the benchmark in a separate thread
        benchmark_process = Process(
            target=lambda: run_benchmark(
                config=agbenchmark_config,
                tests=(body.test,),
                mock=body.mock or False,
            )
        )
        benchmark_process.start()

        # Wait for the benchmark to finish, with a timeout of 200 seconds
        timeout = 200
        start_time = time.time()
        while benchmark_process.is_alive():
            if time.time() - start_time > timeout:
                logger.warning(f"Benchmark run timed out after {timeout} seconds")
                benchmark_process.terminate()
                break
            time.sleep(1)
        else:
            logger.debug(f"Benchmark finished running in {time.time() - start_time} s")

        # List all folders in the current working directory
        reports_folder = agbenchmark_config.reports_folder
        folders = [folder for folder in reports_folder.iterdir() if folder.is_dir()]

        # Sort the folders based on their names
        sorted_folders = sorted(folders, key=lambda x: x.name)

        # Get the last folder
        latest_folder = sorted_folders[-1] if sorted_folders else None

        # Read report.json from this folder
        if latest_folder:
            report_path = latest_folder / "report.json"
            logger.debug(f"Getting latest report from {report_path}")
            if report_path.exists():
                with report_path.open() as file:
                    data = json.load(file)
                logger.debug(f"Report data: {data}")
            else:
                raise HTTPException(
                    502,
                    "Could not get result after running benchmark: "
                    f"'report.json' does not exist in '{latest_folder}'",
                )
        else:
            raise HTTPException(
                504, "Could not get result after running benchmark: no reports found"
            )

        return data

    @router.post("/agent/tasks", tags=["agent"])
    async def create_agent_task(task_eval_request: TaskEvalRequestBody) -> Task:
        """
        Creates a new task using the provided TaskEvalRequestBody and returns a Task.

        Args:
            task_eval_request: `TaskRequestBody` including an eval_id.

        Returns:
            Task: A new task with task_id, input, additional_input,
                and empty lists for artifacts and steps.

        Example:
            Request (TaskEvalRequestBody defined in schema.py):
                {
                    ...,
                    "eval_id": "50da533e-3904-4401-8a07-c49adf88b5eb"
                }

            Response (Task defined in `agent_protocol_client.models`):
                {
                    "task_id": "50da533e-3904-4401-8a07-c49adf88b5eb",
                    "input": "Write the word 'Washington' to a .txt file",
                    "artifacts": []
                }
        """
        try:
            challenge_info = CHALLENGES[task_eval_request.eval_id]
            async with ApiClient(configuration) as api_client:
                api_instance = AgentApi(api_client)
                task_input = challenge_info.task

                task_request_body = TaskRequestBody(
                    input=task_input, additional_input=None
                )
                task_response = await api_instance.create_agent_task(
                    task_request_body=task_request_body
                )
                task_info = BenchmarkTaskInfo(
                    task_id=task_response.task_id,
                    start_time=datetime.datetime.now(datetime.timezone.utc),
                    challenge_info=challenge_info,
                )
                task_informations[task_info.task_id] = task_info

                if input_artifacts_dir := challenge_info.task_artifacts_dir:
                    await upload_artifacts(
                        api_instance,
                        input_artifacts_dir,
                        task_response.task_id,
                        "artifacts_in",
                    )
                return task_response
        except ApiException as e:
            logger.error(f"Error whilst trying to create a task:\n{e}")
            logger.error(
                "The above error was caused while processing request: "
                f"{task_eval_request}"
            )
            raise HTTPException(500)

    @router.post("/agent/tasks/{task_id}/steps")
    async def proxy(request: Request, task_id: str):
        timeout = httpx.Timeout(300.0, read=300.0)  # 5 minutes
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Construct the new URL
            new_url = f"{configuration.host}/ap/v1/agent/tasks/{task_id}/steps"

            # Forward the request
            response = await client.post(
                new_url,
                content=await request.body(),
                headers=dict(request.headers),
            )

            # Return the response from the forwarded request
            return Response(content=response.content, status_code=response.status_code)

    @router.post("/agent/tasks/{task_id}/evaluations")
    async def create_evaluation(task_id: str) -> BenchmarkRun:
        task_info = task_informations[task_id]
        challenge = get_challenge_from_source_uri(task_info.challenge_info.source_uri)
        try:
            async with ApiClient(configuration) as api_client:
                api_instance = AgentApi(api_client)
                eval_results = await challenge.evaluate_task_state(
                    api_instance, task_id
                )

            eval_info = BenchmarkRun(
                repository_info=RepositoryInfo(),
                run_details=RunDetails(
                    command=f"agbenchmark --test={challenge.info.name}",
                    benchmark_start_time=(
                        task_info.start_time.strftime("%Y-%m-%dT%H:%M:%S+00:00")
                    ),
                    test_name=challenge.info.name,
                ),
                task_info=TaskInfo(
                    data_path=challenge.info.source_uri,
                    is_regression=None,
                    category=[c.value for c in challenge.info.category],
                    task=challenge.info.task,
                    answer=challenge.info.reference_answer or "",
                    description=challenge.info.description or "",
                ),
                metrics=Metrics(
                    success=all(e.passed for e in eval_results),
                    success_percentage=(
                        100 * sum(e.score for e in eval_results) / len(eval_results)
                        if eval_results  # avoid division by 0
                        else 0
                    ),
                    attempted=True,
                ),
                config={},
            )

            logger.debug(
                f"Returning evaluation data:\n{eval_info.model_dump_json(indent=4)}"
            )
            return eval_info
        except ApiException as e:
            logger.error(f"Error {e} whilst trying to evaluate task: {task_id}")
            raise HTTPException(500)

    app.include_router(router, prefix="/ap/v1")

    return app
