from agent_protocol import Agent, Step, Task
import time 
import autogpt.utils


class AutoGPT:


    def __init__(self) -> None:
        pass

    async def task_handler(self, task: Task) -> None:
        print(f"task: {task.input}")
        await Agent.db.create_step(task.task_id, task.input, is_last=True)
        time.sleep(2)
        autogpt.utils.run(task.input)
        # print(f"Created Task id: {task.task_id}")
        return task
        

    async def step_handler(self, step: Step) -> Step:
        # print(f"step: {step}")
        agent_step = await Agent.db.get_step(step.task_id, step.step_id)
        updated_step: Step = await Agent.db.update_step(agent_step.task_id, agent_step.step_id, status="completed")
        updated_step.output = agent_step.input
        print(f"Step completed: {updated_step}")
        return updated_step
