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

MODEL_NAME = "gpt-4"  # gpt-3.5-turbo, gpt-4

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

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        LOG.info("📦 Executing step")
        task = await self.db.get_task(task_id)

        step, ability = await self.create_step(task, step_request)

        LOG.info(f"Run ability {ability['name']} with arguments {ability['args']}")

        try:
            output = await self.abilities.run_ability(
                task_id, ability["name"], **ability["args"]
            )
        except Exception as e:
            failure = f"Failed to run ability {ability['name']} with arguments {ability['args']}: {str(e)}"
            LOG.warning(f"Step failed: {step.step_id}. {failure}")
            await self.db.update_step(task.task_id, step.step_id, "completed", output=failure)
            return step

        # FIXME: Just to speed up the agent
        if ability["name"] in ["write_code", "fix_code"]:
            success, output = output
            if success:
                LOG.debug(f"Will set is_last because tests passed")
                step.is_last = True

        step.output = str(output)
        LOG.debug(f"Executed step [{step.name}] output:\n{step.output}")

        await self.db.update_step(task.task_id, step.step_id, "completed", output=step.output, is_last=step.is_last)
        LOG.info(f"Step completed: {step.step_id} input: {step.input[:19]}")

        if step.is_last:
            LOG.info(f"Task completed: {task.task_id} input: {task.input[:19]}")

        return step

    async def create_step(self, task: Task, step_request: StepRequestBody):
        #prompt_engine = PromptEngine("create-step")
        prompt_engine = PromptEngine("create-step-with-reasoning")

        previous_steps, page = await self.db.list_steps(task.task_id, per_page=100)
        if len(previous_steps) > 5:  # FIXME: To not end up in infinite test improvement loop
            ability = {
                "name": "finish",
                "args": {
                    "reason": "Giving up..."
                }
            }
            step_request.name = "Giving up"
            step_request.input = "Giving up"
            step = await self.db.create_step(
                task_id=task.task_id,
                input=step_request,
                is_last=True,
                additional_input={"ability": ability}
            )
            return step, ability
        if previous_steps:
            last_step = previous_steps[-1]
            LOG.info(f"Found {len(previous_steps)} previously executed steps. Last executed step was: {last_step.name}.")

            # FIXME: Just to speed up the agent
            if ("ability" in last_step.additional_input and
                    last_step.additional_input["ability"]["name"] in ["write_code", "fix_code"]):

                last_ability = last_step.additional_input["ability"]
                step_request.name = "Fix code"
                step_request.input = last_step.output
                ability = {
                    "name": "fix_code",
                    "args": {
                        "instructions": last_step.output,
                        "file": last_ability["args"]["file"],
                        "test_file": last_ability["args"]["test_file"]
                    }
                }
                step = await self.db.create_step(
                    task_id=task.task_id,
                    input=step_request,
                    is_last=False,
                    additional_input={"ability": ability}
                )

                return step, ability

            previous_steps = previous_steps[:-1]
        else:
            LOG.info(f"No previous steps found.")
            last_step = None

        task_kwargs = {
            "abilities": self.abilities.list_abilities_for_prompt()
        }

        system_prompt = prompt_engine.load_prompt("system-prompt", **task_kwargs)
        system_format = prompt_engine.load_prompt("step-format")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": system_format},
        ]

        task_kwargs = {
            "task": task.input,
            "step": last_step,
            "previous_steps": previous_steps
        }

        task_prompt = prompt_engine.load_prompt("user-prompt", **task_kwargs)
        messages.append({"role": "user", "content": task_prompt})

        LOG.info("User: " + task_prompt)

        answer = await self.do_steps_request(messages)

        step_request.name = answer["step"]["name"]

        if "thoughts" in answer and "speak" in answer["thoughts"]:
            step_request.input = answer["thoughts"]["speak"]

        step = await self.db.create_step(
            task_id=task.task_id,
            input=step_request,
            is_last=answer["step"]["ability"]["name"] == "finish",
            additional_input={"ability": answer["step"]["ability"]}
        )

        return step, answer["step"]["ability"]

    async def do_steps_request(self, messages: List[dict], retry: int = 0):
        chat_completion_kwargs = {
            "messages": messages,
            "model": MODEL_NAME,
        }

        async def do_retry(retry_messages: List[dict]):
            if retry < 2:
                messages.extend(retry_messages)
                return await self.do_steps_request(messages, retry=retry + 1)
            else:
                LOG.info(f"Retry limit reached, aborting")
                raise Exception("Failed to create steps")

        try:
            #LOG.info(pprint.pformat(messages))
            chat_response = await chat_completion_request(**chat_completion_kwargs)
            response = chat_response["choices"][0]["message"]["content"]
            answer = json.loads(chat_response["choices"][0]["message"]["content"])
            LOG.info(pprint.pformat(answer))
        except json.JSONDecodeError as e:
            LOG.warning(f"Unable to parse chat response: {response}. Got exception {e}")
            return await do_retry([{"role": "user", "content": f"Invalid response. {e}. Please try again."}])
        except Exception as e:
            LOG.error(f"Unable to generate chat response: {e}")
            raise e

        if "step" not in answer and not answer["step"] and not isinstance(answer["step"], dict):
            LOG.info(f"No step provided, retry {retry}")
            return await do_retry([{"role": "user", "content": "You must provide a step."}])

        invalid_abilities = self.validate_ability(answer["step"])
        if invalid_abilities:
            return await do_retry(messages)

        if "thoughts" in answer and answer["thoughts"]:
            LOG.debug(f"Thoughts:")
            if "reasoning" in answer["thoughts"]:
                LOG.debug(f"\tReasoning: {answer['thoughts']['reasoning']}")
            if "criticism" in answer["thoughts"]:
                LOG.debug(f"\tCriticism: {answer['thoughts']['criticism']}")
            if "text" in answer["thoughts"]:
                LOG.debug(f"\tText: {answer['thoughts']['text']}")
            if "speak" in answer["thoughts"]:
                LOG.debug(f"\tSpeak: {answer['thoughts']['speak']}")
        else:
            LOG.info(f"No thoughts provided")

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
