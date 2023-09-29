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
        task = await self.db.get_task(task_id)

        # Create a new step in the database
        # have AI determine last step
        step = await self.db.create_step(
            task_id=task_id, input=step_request, is_last=False
        )

        # setup chatcompletion to achieve this task
        # with custom prompt that will generate steps
        prompt_engine = PromptEngine("gpt-3.5-turbo")

        # set up reply json with alternative created
        system_prompt = prompt_engine.load_prompt("alt-system-prompt")

        # add to messages
        self.messages.append(
            {"role": "system", "content": system_prompt}
        )
        
        # chat initialize if no steps else load past
        # steps into prompt

        list_steps = await self.db.list_steps(task_id)
        past_steps = []

        if len(list_steps[0]) > 0:
            for past_steps in list_steps:
                past_steps.append(
                    (past_steps.input, past_steps.output))
        
        ontology_prompt_params = {
            "task": task.input,
            "request": step.input,
            "abilities": self.abilities.list_abilities_for_prompt(),
            "past_steps": str(past_steps)
        }

        task_prompt = prompt_engine.load_prompt(
            "ontology-format",
            **ontology_prompt_params
        )
        
        self.messages.append({"role": "user", "content": task_prompt})

        try:
            chat_completion_parms = {
                "messages": self.messages,
                "model": "gpt-3.5-turbo"
            }

            chat_response = await chat_completion_request(
                **chat_completion_parms)
            answer = json.loads(
                chat_response["choices"][0]["messages"]["content"])
            
            LOG.info(pprint.pformat(answer))

        except json.JSONDecodeError as e:
            # Handle JSON decoding errors
            LOG.error(f"📢 Unable to decode chat response: {chat_response}")
        except Exception as e:
            # Handle other exceptions
            LOG.error(f"📢 Unable to generate chat response: {e}")

        # Extract the ability from the answer
        ability = answer["step"]["ability"]

        LOG.info(f"🔨 Running Ability {ability}")

        # Run the ability and get the output
        output = await self.abilities.run_ability(
            task_id, ability["name"], **ability["args"]
        )

        LOG.info(f"🔨 Output: {output}")

        # Set the step output and is_last from AI
        step.output = answer["step"]["reason"]
        step.is_last = answer["step"]["is_last_step"]

        # Return the completed step
        return step
