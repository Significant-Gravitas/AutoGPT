import json
import pprint
import uuid

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

from forge.sdk.memory.memstore import ChromaMemStore

LOG = ForgeLogger(__name__)


class ForgeAgent(Agent):
    def __init__(self, database: AgentDB, workspace: Workspace):
        super().__init__(database, workspace)

        # create a uuid
        self.uuid = str(uuid.uuid4())

        # initialize memstore
        self.memory = ChromaMemStore(f".{self.uuid}")

        # initialize local message store
        self.messages = []
    
    async def create_task(self, task_request: TaskRequestBody) -> Task:
        try:
            task = await self.create_task(task_request)

            LOG.info(
                f"📦 Task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
            )

            # add task information to the memstore
            memory_metadata = {
                "created": task.created_at,
                "modified": task.modified_at
            }
            
            self.memory.add(task.task_id, task.input, memory_metadata)

            LOG.info(
                f"🧠 Task added to memory "
            )
        except Exception as err:
            LOG.error(f"📢 create_task failed: {err}")
            raise err

        return task

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        try:
            task = await self.db.get_task(task_id)

            # would like to change this so steps are AI driven
            # use task to create steps

            # setup chatcompletion to achieve this task
            # with custom prompt that will generate steps
            prompt_engine = PromptEngine("gpt-3.5-turbo")

            # set up reply json with alternative created
            system_prompt = prompt_engine.load_prompt("alt-system-prompt")
            
            # chat initialize if no steps else load past
            # steps into prompt
            ontology_prompt_params = {
                "task": task.input,
                "abilities": self.abilities.list_abilities_for_prompt(),
                "past_steps": "None"
            }

            initial_task_prompt = prompt_engine.load_prompt(
                "ontology-format",
                **ontology_prompt_params
            )
            
        except Exception as err:
            LOG.error(f"📢 execute_step failed: {err}")
            raise err
