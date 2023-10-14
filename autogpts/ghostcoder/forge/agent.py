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

planning_mode = False
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

    async def execute_step(self, task_id: str, step_request: StepRequestBody, is_retry: bool = False) -> Step:
        LOG.info("📦 Executing step")
        task = await self.db.get_task(task_id)

        ability = await self.select_ability(task)

        step = await self.db.create_step(
            task_id=task_id,
            input=step_request,
            is_last=ability["name"] == "finish",
            additional_input={"ability": ability}
        )

        LOG.info(f"Run ability {ability['name']} with arguments {ability['args']}")
        is_last, output = await self.abilities.run_ability(
            task_id, ability["name"], **ability["args"]
        )

        step.output = str(output)
        LOG.debug(f"Executed step [{step.name}] output:\n{step.output}")

        await self.db.update_step(task.task_id, step.step_id, "completed", output=step.output)
        LOG.info(f"Step completed: {step.step_id} input: {step.input[:19]}")

        if step.is_last:
            LOG.info(f"Task completed: {task.task_id} input: {task.input[:19]}")

        return step

    async def select_ability(self, task: Task):
        previous_steps, page = await self.db.list_steps(task, per_page=100)
        if previous_steps:
            last_step = previous_steps[-1]
        else:
            last_step = None

        prompt_engine = PromptEngine("select-ability")
        task_kwargs = {
            "abilities": self.abilities.list_abilities_for_prompt()
        }
        system_prompt = prompt_engine.load_prompt("system-prompt",  **task_kwargs)
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        task_kwargs = {
            "task": task.input,
            "last_step": last_step,
            "previous_steps": previous_steps
        }

        task_prompt = prompt_engine.load_prompt("user-prompt",  **task_kwargs)
        messages.append({"role": "user", "content": task_prompt})

        chat_completion_kwargs = {
            "messages": messages,
            "model": MODEL_NAME,
        }

        try:
            LOG.debug(pprint.pformat(messages))
            chat_response = await chat_completion_request(**chat_completion_kwargs)
            LOG.debug(pprint.pformat(chat_response["choices"][0]["message"]["content"]))

            ability = json.loads(chat_response["choices"][0]["message"]["content"])
            ability_names = [a.name for a in self.abilities.list_abilities().values()]
            if not isinstance(ability, dict) and not ability["name"] in ability_names:
                LOG.warning(f"Invalid ability: {ability}")

        except json.JSONDecodeError as e:
            LOG.warning(f"Unable to parse chat response: {chat_response}. Error: {e}.")
        except Exception as e:
            LOG.error(f"Unable to generate chat response: {e}")

        return ability

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
