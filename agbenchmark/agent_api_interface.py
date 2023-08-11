import os
import time
from typing import Any, Dict

from agent_protocol_client import AgentApi, ApiClient, Configuration, TaskRequestBody

from agbenchmark.agent_interface import get_list_of_file_paths
from agbenchmark.utils.data_types import ChallengeData


async def run_api_agent(
    task: ChallengeData, config: Dict[str, Any], artifacts_location: str, timeout: int
) -> None:
    configuration = Configuration(host=config["host"])
    async with ApiClient(configuration) as api_client:
        api_instance = AgentApi(api_client)
        task_request_body = TaskRequestBody(input=task.task)

        start_time = time.time()
        response = await api_instance.create_agent_task(
            task_request_body=task_request_body
        )
        task_id = response.task_id

        for file in get_list_of_file_paths(artifacts_location, "artifacts_in"):
            print(f"[{task.name}] - Copy {file.split('/')[-1]} to agent")
            await api_instance.upload_agent_task_artifacts(task_id=task_id, file=file)

        i = 1
        while step := await api_instance.execute_agent_task_step(task_id=task_id):
            print(f"[{task.name}] - step {step.name} ({i}. request)")
            i += 1

            if step.is_last:
                break

            if time.time() - start_time > timeout:
                raise TimeoutError("Time limit exceeded")

        artifacts = await api_instance.list_agent_task_artifacts(task_id=task_id)
        for artifact in artifacts:
            print(f"[{task.name}] - Copy {artifact.file_name} from agent")

            if artifact.relative_path:
                folder_path = os.path.join(config["workspace"], artifact.relative_path)
            else:
                folder_path = os.path.join(config["workspace"])

            with open(os.path.join(folder_path, artifact.file_name), "wb") as f:
                content = await api_instance.download_agent_task_artifact(
                    task_id=task_id, artifact_id=artifact.artifact_id
                )
                f.write(content)
