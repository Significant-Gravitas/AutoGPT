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
    chat_completion_request,
)

LOG = ForgeLogger(__name__)

    class ForgeAgent(Agent):

    prompt_engine = PromptEngine("gpt-3.5-turbo")
    system_prompt = prompt_engine.load_prompt("system-format")


    task_kwargs = {
        "task": task.input,
        "abilities": self.abilities.list_abilities_for_prompt(),
    }

    task_prompt = prompt_engine.load_prompt("task-step", **task_kwargs)

    messages list:
    messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task_prompt}
            ]

    try:
    # Define the parameters for the chat completion request
    chat_completion_kwargs = {
        "messages": messages,
        "model": "gpt-3.5-turbo",
    }
    # Make the chat completion request and parse the response
    chat_response = await chat_completion_request(**chat_completion_kwargs)
    answer = json.loads(chat_response["choices"][0]["message"]["content"])

    # Log the answer for debugging purposes
    LOG.info(pprint.pformat(answer))

    except json.JSONDecodeError as e:
        # Handle JSON decoding errors
        LOG.error(f"Unable to decode chat response: {chat_response}")
    except Exception as e:
        # Handle other exceptions
        LOG.error(f"Unable to generate chat response: {e}")


__init__(self, database: AgentDB, workspace: Workspace):
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
        """
        The agent protocol, which is the core of the Forge, works by creating a task and then
        executing steps for that task. This method is called when the agent is asked to execute
        a step.

        The task that is created contains an input string, for the bechmarks this is the task
        the agent has been asked to solve and additional input, which is a dictionary and
        could contain anything.

        If you want to get the task use:

        ```
        task = await self.db.get_task(task_id)
        ```

        The step request body is essentailly the same as the task request and contains an input
        string, for the bechmarks this is the task the agent has been asked to solve and
        additional input, which is a dictionary and could contain anything.

        You need to implement logic that will take in this step input and output the completed step
        as a step object. You can do everything in a single step or you can break it down into
        multiple steps. Returning a request to continue in the step output, the user can then decide
        if they want the agent to continue or not.
        """
        # An example that
        step = await self.db.create_step(
            task_id=task_id, input=step_request, is_last=True
        )

        self.workspace.write(task_id=task_id, path="output.txt", data=b"Washington D.C")


        await self.db.create_artifact(
            task_id=task_id,
            step_id=step.step_id,
            file_name="output.txt",
            relative_path="",
            agent_created=True,
        )
        
        step.output = "Washington D.C"

        LOG.info(f"\t✅ Final Step completed: {step.step_id}")

        return step
