import json
import pprint

from forge.sdk import (
    Agent,
    AgentDB,
    Step,
    StepRequestBody,
    Workspace,
    ForgeLogger,
    Task,
    TaskRequestBody,
    PromptEngine,
    chat_completion_request, Status,
)
from ghostcoder import FileRepository
from ghostcoder.actions import CodeWriter
from ghostcoder.benchmark.utils import create_openai_client
from ghostcoder.schema import Message, TextItem

LOG = ForgeLogger(__name__)


class ForgeAgent(Agent):
    """
    The goal of the Forge is to take care of the boilerplate code so you can focus on
    agent design.

    There is a great paper surveying the agent landscape: https://arxiv.org/abs/2308.11432
    Which I would highly recommend reading as it will help you understand the possabilities.

    Here is a summary of the key components of an agent:

    Anatomy of an agent:
         - Profile
         - Memory
         - Planning
         - Action

    Profile:

    Agents typically perform a task by assuming specific roles. For example, a teacher,
    a coder, a planner etc. In using the profile in the llm prompt it has been shown to
    improve the quality of the output. https://arxiv.org/abs/2305.14688

    Additionally baed on the profile selected, the agent could be configured to use a
    different llm. The possabilities are endless and the profile can be selected selected
    dynamically based on the task at hand.

    Memory:

    Memory is critical for the agent to acculmulate experiences, self-evolve, and behave
    in a more consistent, reasonable, and effective manner. There are many approaches to
    memory. However, some thoughts: there is long term and short term or working memory.
    You may want different approaches for each. There has also been work exploring the
    idea of memory reflection, which is the ability to assess its memories and re-evaluate
    them. For example, condensting short term memories into long term memories.

    Planning:

    When humans face a complex task, they first break it down into simple subtasks and then
    solve each subtask one by one. The planning module empowers LLM-based agents with the ability
    to think and plan for solving complex tasks, which makes the agent more comprehensive,
    powerful, and reliable. The two key methods to consider are: Planning with feedback and planning
    without feedback.

    Action:

    Actions translate the agents decisions into specific outcomes. For example, if the agent
    decides to write a file, the action would be to write the file. There are many approaches you
    could implement actions.

    The Forge has a basic module for each of these areas. However, you are free to implement your own.
    This is just a starting point.
    """

    def __init__(self, database: AgentDB, workspace: Workspace):
        """
        The database is used to store tasks, steps and artifact metadata. The workspace is used to
        store artifacts. The workspace is a directory on the file system.

        Feel free to create subclasses of the database and workspace to implement your own storage
        """
        super().__init__(database, workspace)

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        """
        The agent protocol, which is the core of the Forge, works by creating a task and then
        executing steps for that task. This method is called when the agent is asked to create
        a task.

        We are hooking into function to add a custom log message. Though you can do anything you
        want here.
        """
        task = await super().create_task(task_request)
        LOG.info(
            f"📦 Task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
        )

        return task

    async def plan_steps(self, task, step_request: StepRequestBody):
        step_request.name = "Plan steps"

        step = await self.db.create_step(
            task_id=task.task_id, input=step_request, is_last=False
        )

        prompt_engine = PromptEngine("gpt-3.5-turbo")

        task_kwargs = {
            "abilities": self.abilities.list_abilities_for_prompt(),
        }

        system_prompt = prompt_engine.load_prompt("plan-format",  **task_kwargs)

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        task_kwargs = {
            "task": task.input,
        }

        task_prompt = prompt_engine.load_prompt("plan-task-step",  **task_kwargs)
        messages.append({"role": "user", "content": task_prompt})

        try:
            chat_completion_kwargs = {
                "messages": messages,
                "model": "gpt-4",
            }

            LOG.info(pprint.pformat(messages))

            chat_response = await chat_completion_request(**chat_completion_kwargs)
            answer = json.loads(chat_response["choices"][0]["message"]["content"])

            LOG.info(pprint.pformat(answer))

            step.output = answer["thoughts"]["text"]

            for i, new_step in enumerate(answer["steps"]):
                step_request = StepRequestBody(
                    name=new_step["name"],
                    input=new_step["description"],
                )
                await self.db.create_step(
                    task_id=task.task_id,
                    input=step_request,
                    additional_input={"ability": new_step["ability"]},
                    is_last=i == len(answer["steps"]) - 1
                )

            await self.db.update_step(task.task_id, step.step_id, "completed", output=step.output)
        except json.JSONDecodeError as e:
            LOG.error(f"Unable to decode chat response. Got error {e}. \nThe response was:\n{chat_response}")
            raise e
        except Exception as e:
            LOG.error(f"Unable to generate chat response: {e}")
            raise e

        return step

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        task = await self.db.get_task(task_id)

        steps, page = await self.db.list_steps(task_id)
        if not steps:
            return await self.plan_steps(task, step_request)

        previous_steps = []
        next_steps = []
        for step in steps:
            if step.status == Status.created:
                next_steps.append(step)
            else:
                previous_steps.append(step)

        if not next_steps:
            raise ValueError("No next step")

        prompt_engine = PromptEngine("gpt-3.5-turbo")

        system_prompt = prompt_engine.load_prompt("system-format")

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        next_step_dicts = [{"name": step.name,
                            "description": step.input,
                            "ability": step.additional_input["ability"]
                           } for step in next_steps[1:]]
        if next_step_dicts:
            next_steps_json = json.dumps(next_step_dicts)
        else:
            next_steps_json = None

        current_step = next_steps[0]
        ability = current_step.additional_input["ability"]

        task_kwargs = {
            "task": task.input,
            "step": current_step.input,
            "ability": json.dumps(ability),
            "next_steps": next_steps_json,
            "abilities": self.abilities.list_abilities_for_prompt(),
            "previous_steps": previous_steps
        }

        task_prompt = prompt_engine.load_prompt("task-step", **task_kwargs)

        messages.append({"role": "user", "content": task_prompt})

        try:
            chat_completion_kwargs = {
                "messages": messages,
                "model": "gpt-4",
            }

            #LOG.info(pprint.pformat(messages))

            chat_response = await chat_completion_request(**chat_completion_kwargs)
            answer = json.loads(chat_response["choices"][0]["message"]["content"])

            print(answer["thoughts"])

            # Log the answer for debugging purposes
            #LOG.info(pprint.pformat(answer))

        except json.JSONDecodeError as e:
            LOG.error(f"Unable to decode chat response. Got error {e}. \nThe response was:\n{chat_response}")
            raise e

        except Exception as e:
            LOG.error(f"Unable to generate chat response: {e}")
            raise e

        if "ability" in answer:
            if (isinstance(answer["ability"], dict) and answer["ability"] and
                    answer["ability"]["name"] in [a.name for a in self.abilities.list_abilities()]):
                LOG.info(f"\tReplace ability {ability['name']} with {answer['ability']['name']}")
                ability = answer["ability"]
            else:
                LOG.info(f"Invalid ability provided {answer['ability']}")

        if "next_steps" in answer and answer["next_steps"]:
            if not isinstance(answer["next_steps"], list):
                LOG.info(f"Invalid next steps provided {answer['next_steps']}")
            else:
                LOG.info(f"\tReplace {len(next_steps)} steps with {len(answer['next_steps'])} new steps")
                for next_step in next_steps:
                    await self.db.update_step(task.task_id, next_step.step_id, "skipped")

                for i, new_step in enumerate(answer["next_steps"]):
                    LOG.info(f"\t{i} {new_step['name']}:\n{new_step['description']}\n{new_step['ability']}")
                    step_request = StepRequestBody(
                        name=new_step["name"],
                        input=new_step["description"],
                    )
                    await self.db.create_step(
                        task_id=task.task_id,
                        input=step_request,
                        additional_input={"ability": new_step["ability"]},
                        is_last=i == len(answer["next_steps"]) - 1
                    )

        if ability["name"] == "finish":
            LOG.info(f"\tFinish task")
            current_step.is_last = True
            current_step.output = answer["thoughts"]["speak"]
        else:
            LOG.info(f"\tRun ability {ability}")
            output = await self.abilities.run_ability(
                task_id, ability["name"], **ability["args"]
            )

            current_step.output = output

        await self.db.update_step(task.task_id, current_step.step_id, "completed", output=current_step.output)

        LOG.info(f"\t✅ Step completed: {current_step.step_id} input: {current_step.input[:19]}")

        return current_step
