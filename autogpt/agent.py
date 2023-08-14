from agent_protocol import Agent, Step, Task

import autogpt.utils


class AutoGPT:


    def __init__(self) -> None:
        pass

    async def task_handler(self, task: Task) -> None:
        print(f"task: {task.input}")
        autogpt.utils.run(task.input)
        await Agent.db.create_step(task.task_id, task.input)

    async def step_handler(self, step: Step) -> Step:
        print(f"step: {step.input}")
        step = Agent.db.get_step(step.step_id)
        updated_step: Step = Agent.db.update_step(step.step_id, status="completed")
        updated_step.output = step.input
        return updated_step
