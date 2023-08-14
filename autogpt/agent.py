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
        await Agent.db.create_step(
            step.task_id, f"Nothing to see here.. {step.name}", is_last=True
        )
        step.output = step.input
        return step
