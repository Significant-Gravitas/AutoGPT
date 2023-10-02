import json
import pprint
from typing import List

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

LOG = ForgeLogger(__name__)

MODEL_NAME = "gpt-4"  # gpt-3.5-turbo



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

        if not step_request.input:
            step_request.input = "Create steps to accomplish the objective"

        step = await self.db.create_step(
            task_id=task.task_id, input=step_request, is_last=False
        )

        files = self.workspace.list(task.task_id, "/")

        prompt_engine = PromptEngine("plan-steps")
        task_kwargs = {
            "abilities": self.abilities.list_abilities_for_prompt(),
            "files": files
        }
        system_prompt = prompt_engine.load_prompt("system-prompt",  **task_kwargs)
        system_format = prompt_engine.load_prompt("step-format")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": system_format},
        ]

        task_kwargs = {
            "task": task.input,
        }
        task_prompt = prompt_engine.load_prompt("user-prompt",  **task_kwargs)
        messages.append({"role": "user", "content": task_prompt})

        answer = await self.do_steps_request(messages, new_plan=True)

        await self.create_steps(task.task_id, answer["steps"])
        await self.db.update_step(task.task_id, step.step_id, "completed", output=answer["thoughts"]["text"])

        return step

    async def execute_step(self, task_id: str, step_request: StepRequestBody, is_retry: bool = False) -> Step:
        task = await self.db.get_task(task_id)

        steps, page = await self.db.list_steps(task_id)
        if not steps:
            return await self.plan_steps(task, step_request)

        previous_steps = []
        next_steps = []
        for step in steps:
            if step.status == Status.created:
                next_steps.append(step)
            elif step.status == Status.completed:
                previous_steps.append(step)

        if not next_steps:
            LOG.info(f"Tried to execute with no next steps, return last step as the last")
            step = previous_steps[-1]
            step.is_last = True
            return step

        current_step = next_steps[0]
        next_steps = next_steps[1:]
        ability = current_step.additional_input["ability"]

        prompt_engine = PromptEngine("run-ability")
        system_kwargs = {
            "abilities": self.abilities.list_abilities_for_prompt(),
            "previous_steps": previous_steps
        }
        system_prompt = prompt_engine.load_prompt("system-prompt", **system_kwargs)

        ability_kwargs = {
            "ability": json.dumps(ability),
            "previous_steps": previous_steps
        }
        ability_prompt = prompt_engine.load_prompt("user-prompt", **ability_kwargs)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ability_prompt},
        ]

        chat_completion_kwargs = {
            "messages": messages,
            "model": MODEL_NAME,
        }

        try:
            #LOG.debug(pprint.pformat(messages))
            chat_response = await chat_completion_request(**chat_completion_kwargs)
            ability_answer = json.loads(chat_response["choices"][0]["message"]["content"])
            ability_names = [a.name for a in self.abilities.list_abilities().values()]
            if isinstance(ability_answer, dict) and ability_answer["name"] in ability_names:
                if ability != ability_answer:
                    LOG.info(f"Update ability: {ability_answer}")
                ability = ability_answer
            else:
                LOG.warning(f"Invalid ability: {ability_answer}")

        except json.JSONDecodeError as e:
            LOG.warning(f"Unable to parse chat response: {chat_response}. Error: {e}.")
        except Exception as e:
            LOG.error(f"Unable to generate chat response: {e}")

        if ability["name"] == "finish":
            LOG.info(f"Finish task")
            current_step.is_last = True
        else:
            LOG.info(f"Run ability {ability}")
            output = await self.abilities.run_ability(
                task_id, ability["name"], **ability["args"]
            )

            current_step.output = str(output)

            prompt_engine = PromptEngine("review-steps")

            system_kwargs = {
                "abilities": self.abilities.list_abilities_for_prompt(),
                "files": self.workspace.list(task.task_id, "/")
            }

            system_prompt = prompt_engine.load_prompt("system-prompt", **system_kwargs)
            system_format = prompt_engine.load_prompt("step-format")

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": system_format},
            ]

            LOG.debug("Review next steps:")
            for i, step in enumerate(next_steps):
                LOG.debug(f"{i+1}: {step.name}: {step.input}")

            next_step_dicts = [{"name": step.name,
                                "description": step.input,
                                "ability": step.additional_input["ability"]
                                } for step in next_steps]
            if next_step_dicts:
                next_steps_json = json.dumps(next_step_dicts)
            else:
                next_steps_json = None

            task_kwargs = {
                "task": task.input,
                "step": current_step,
                "next_steps": next_steps_json,
                "previous_steps": previous_steps
            }

            task_prompt = prompt_engine.load_prompt("user-prompt", **task_kwargs)
            messages.append({"role": "user", "content": task_prompt})

            answer = await self.do_steps_request(messages, new_plan=False)

            if "steps" in answer and answer["steps"]:
                if not isinstance(answer["steps"], list):
                    LOG.info(f"Invalid next steps provided {answer['steps']}")
                else:
                    LOG.info(f"Replace {len(next_steps)} steps with {len(answer['steps'])} new steps")
                    for next_step in next_steps:
                        await self.db.update_step(task.task_id, next_step.step_id, "skipped")

                    next_steps = []
                    for i, new_step in enumerate(answer["steps"]):
                        LOG.info(f"Create step {i + 1} {new_step['name']}:\n{new_step['description']}\n{new_step['ability']}")
                        await self.create_step(task.task_id, new_step)
                        next_steps.append(new_step)

        await self.db.update_step(task.task_id, current_step.step_id, "completed", output=current_step.output)

        LOG.info(f"Step completed: {current_step.step_id} input: {current_step.input[:19]}")

        if not next_steps:
            LOG.info(f"Task completed: {task.task_id} input: {task.input[:19]}")
            current_step.is_last = True

        return current_step

    async def do_steps_request(self, messages: List[dict], new_plan: bool = False, retry: int = 0):
        chat_completion_kwargs = {
            "messages": messages,
            "model": MODEL_NAME,
        }
        async def do_retry(retry_messages: List[dict]):
            if retry < 2:
                messages.extend(retry_messages)
                return await self.do_steps_request(messages, new_plan, retry=retry + 1)
            else:
                LOG.info(f"Retry limit reached, aborting")
                raise Exception("Failed to create steps")

        try:
            LOG.debug(pprint.pformat(messages))
            chat_response = await chat_completion_request(**chat_completion_kwargs)
            answer = json.loads(chat_response["choices"][0]["message"]["content"])
            LOG.debug(pprint.pformat(answer))
        except json.JSONDecodeError as e:
            LOG.warning(f"Unable to parse chat response: {e}")
            return await do_retry([{"role": "user", "content": f"Invalid response. {e}. Please try again."}])
        except Exception as e:
            LOG.error(f"Unable to generate chat response: {e}")
            raise e

        if new_plan and "steps" not in answer and not answer["steps"]:
            LOG.info(f"No steps provided, retry {retry}")
            return await do_retry([{"role": "user", "content": "You must provide at least one step."}])

        for step in answer["steps"]:
            invalid_abilities = self.validate_ability(step)
            if invalid_abilities:
                return await do_retry(messages)

        return answer

    def validate_ability(self, step: dict):
        ability_names = [a.name for a in self.abilities.list_abilities().values()]
        invalid_abilities = []
        if "ability" not in step or not step["ability"]:
            invalid_abilities.append(f"No ability found in step {step['name']}")
        elif not isinstance(step["ability"], dict):
            invalid_abilities.append(f"The ability in step {step['name']} was defined as a dictionary")
        elif step["ability"]["name"] not in ability_names:
            invalid_abilities.append(f"Ability {step['ability']['name']} in step {step['name']} does not exist, "
                                     f"valid abilities are: {ability_names}")
        return invalid_abilities

    async def create_steps(self, task_id: str, steps: list[dict]):
        for i, step in enumerate(steps):
            LOG.info(f"Create step {i+1} {step['name']}:\n{step['description']}\n{step['ability']}")
            await self.create_step(task_id, step)

    async def create_step(self, task_id: str, step: dict):
        step_request = StepRequestBody(
            name=step["name"],
            input=step["description"],
        )

        await self.db.create_step(
            task_id=task_id,
            input=step_request,
            additional_input={"ability": step["ability"]}
        )
