import sys
import time
from typing import Any, Dict, Optional

from agbenchmark.__main__ import TEMP_FOLDER_ABS_PATH
from agbenchmark.agent_interface import get_list_of_file_paths
from agbenchmark.utils.data_types import ChallengeData
from agent_protocol_client import AgentApi, ApiClient, Configuration, TaskRequestBody


async def run_api_agent(
    task: ChallengeData, config: Dict[str, Any], artifacts_location: str, timeout: int
) -> None:
    host_value = None

    configuration = Configuration(host=config["AgentBenchmarkConfig"].host)
    async with ApiClient(configuration) as api_client:
        api_instance = AgentApi(api_client)
        task_request_body = TaskRequestBody(input=task.task)

        start_time = time.time()
        response = await api_instance.create_agent_task(
            task_request_body=task_request_body
        )
        task_id = response.task_id

        await upload_artifacts(
            api_instance, artifacts_location, task_id, "artifacts_in"
        )

        i = 1
        steps_remaining = True
        while steps_remaining:
            step = await api_instance.execute_agent_task_step(task_id=task_id)
            print(f"[{task.name}] - step {step.name} ({i}. request)")
            i += 1

            if time.time() - start_time > timeout:
                raise TimeoutError("Time limit exceeded")
            if not step or step.is_last:
                steps_remaining = False
        # if we're calling a mock agent, we "cheat" and give the correct artifacts to pass the tests
        if "--mock" in sys.argv:
            await upload_artifacts(
                api_instance, artifacts_location, task_id, "artifacts_out"
            )

        artifacts = await api_instance.list_agent_task_artifacts(task_id=task_id)
        for artifact in artifacts:
            # current absolute path of the directory of the file
            directory_location = TEMP_FOLDER_ABS_PATH
            if artifact.relative_path:
                directory_location = directory_location / artifact.relative_path

            with open(directory_location / artifact.file_name, "wb") as f:
                content = await api_instance.download_agent_task_artifact(
                    task_id=task_id, artifact_id=artifact.artifact_id
                )

                f.write(content)


async def upload_artifacts(
    api_instance: ApiClient, artifacts_location: str, task_id: str, type: str
) -> None:
    for file_path in get_list_of_file_paths(artifacts_location, type):
        relative_path: Optional[str] = "/".join(
            file_path.split(f"{type}/", 1)[-1].split("/")[:-1]
        )
        if not relative_path:
            relative_path = None

        await api_instance.upload_agent_task_artifacts(
            task_id=task_id, file=file_path, relative_path=relative_path
        )
