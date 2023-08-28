import autogpt.sdk.agent
from autogpt.sdk.schema import Step, StepRequestBody


class AutoGPTAgent(autogpt.sdk.agent.Agent):
    async def create_and_execute_step(
        self, task_id: str, step_request: StepRequestBody
    ) -> Step:
        """
        Create a step for the task and execute it.
        """
        return await super().create_and_execute_step(task_id, step_request)
