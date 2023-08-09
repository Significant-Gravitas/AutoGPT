from agent_protocol import Agent, Step, Task, router
from fastapi import APIRouter, Response

import autogpt.utils

##################################################
##################################################
# E2b boilerplate code
# We need to add heartbeat endpoint to agent_protocol
# so we can detect when the agent server is ready
e2b_extension_router = APIRouter()


@e2b_extension_router.get("/hb")
async def hello():
    return Response("Agent running")


e2b_extension_router.include_router(router)


def start_agent(port: int):
    Agent.setup_agent(task_handler, step_handler).start(
        port=port, router=e2b_extension_router
    )


###################################################
###################################################


async def task_handler(task: Task) -> None:
    print(f"task: {task.input}")
    autogpt.utils.run(task.input)
    await Agent.db.create_step(task.task_id, task.input)


async def step_handler(step: Step) -> Step:
    print(f"step: {step.input}")
    await Agent.db.create_step(
        step.task_id, f"Nothing to see here.. {step.name}", is_last=True
    )
    step.output = step.input
    return step
