import json
import pprint
import uuid
import os

from datetime import datetime

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
    ProfileGenerator
)

from forge.sdk.memory.memstore import ChromaMemStore

LOG = ForgeLogger(__name__)


class ForgeAgent(Agent):
    def __init__(self, database: AgentDB, workspace: Workspace):
        super().__init__(database, workspace)
        
        # initialize chat history per uuid
        self.chat_history = {}

        # memory storage
        self.memory = None

    def add_chat(self, task_id: str, role: str, content: str):
        chat_struct = {"role": role, "content": content}
        try:
            if chat_struct not in self.chat_history[task_id]:
                self.chat_history[task_id].append(chat_struct)
        except KeyError:
            self.chat_history[task_id] = [chat_struct]
    
    async def create_task(self, task_request: TaskRequestBody) -> Task:
        try:
            task = await self.db.create_task(
                input=task_request.input,
                additional_input=task_request.additional_input
            )

            LOG.info(
                f"📦 Task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
            )
        except Exception as err:
            LOG.error(f"create_task failed: {err}")
            raise err
        
        # initalize memstore for task
        try:
            # initialize memstore
            self.memory = ChromaMemStore(
                f"{os.getenv('AGENT_WORKSPACE')}/{task.task_id}")
            LOG.info(f"🧠 Created memorystore @ {os.getenv('AGENT_WORKSPACE')}/{task.task_id}")
        except Exception as err:
            LOG.error(f"memstore creation failed: {err}")
            raise err

        return task

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        task = await self.db.get_task(task_id)
        
        # get past steps if any
        # this currently not helping so will need to use the chat
        # list_steps = await self.db.list_steps(task_id)
        # past_step = ""
        # if len(list_steps[0]) > 0:
        #     # get the last recent step to place in prompt memory
        #     # sort steps and get the most recent one
        #     past_step = sorted(list_steps[0], key=lambda x: x.modified_at)[0]

        # Create a new step in the database
        # have AI determine last step
        step = await self.db.create_step(
            task_id=task_id,
            input=step_request,
            additional_input=step_request.additional_input,
            is_last=False
        )

        # setup chatcompletion to achieve this task
        # with custom prompt that will generate steps
        prompt_engine = PromptEngine("gpt-3.5-turbo")

        # set up reply json with alternative created
        system_prompt = prompt_engine.load_prompt("system-format-last")

        # add to messages
        # wont memory store this as static
        self.add_chat(task_id, "system", system_prompt)

        # use AI to get the proper role experts for the task
        profile_gen = ProfileGenerator(task=task)

        print(profile_gen.role_find())
        
        ontology_prompt_params = {
            "role_expert": "Testing",
            "task": task.input,
            "abilities": self.abilities.list_abilities_for_prompt()
        }

        task_prompt = prompt_engine.load_prompt(
            "ontology-format",
            **ontology_prompt_params
        )

        # find memories similar to prompt
        past_chat = self.memory.query(
            task_id=task_id,
            query=task_prompt,
            filters={"role": "assistant"}
        )

        # if memory found add past-convo prompt to chat
        if len(past_chat["documents"][0]) > 0:
            past_convo_params = {
                "previous_chat": past_chat["documents"][0][:1]
            }

            past_convo_prompt = prompt_engine.load_prompt(
                "past-convo",
                **past_convo_params
            )

            self.add_chat(task_id, "user", past_convo_prompt)

            LOG.info(f"🧠 added past convo {pprint.pformat(past_convo_prompt)}")

        # add task to memory store
        self.memory.add(
            task_id=task_id,
            document=task_prompt,
            metadatas={
                "role": "user"
            }
        )

        self.add_chat(task_id, "user", task_prompt)

        try:
            chat_completion_parms = {
                "messages": self.chat_history[task_id],
                "model": "gpt-3.5-turbo"
            }

            LOG.info(f"chat log\n{pprint.pformat(self.chat_history[task_id])}")

            chat_response = await chat_completion_request(
                **chat_completion_parms)
            
            answer = json.loads(
                chat_response["choices"][0]["message"]["content"])
            
            LOG.info(pprint.pformat(answer))

            # add to memory
            self.memory.add(
                task_id=task_id,
                document=chat_response["choices"][0]["message"]["content"],
                metadatas={"role": "assistant"}
            )

            # Extract the ability from the answer
            ability = answer["ability"]

            LOG.info(f"🔨 Running Ability {ability}")

            # Run the ability and get the output
            if "args" in ability:
                ability_args = ability["args"]
            else:
                ability_args = None

            if "args" in ability:
                output = await self.abilities.run_ability(
                    task_id,
                    ability["name"],
                    **ability["args"]
                )
            else:
                output = await self.abilities.run_ability(
                    task_id,
                    ability["name"]
                )

            output = str(output) if output else "Success"

            # add to converstion
            ability_json = {
                "ability": {
                    "name": ability["name"],
                    "args": ability_args
                },
                "output": output
            }

            ability_str = f"ability {ability['name']} ran with paramters {ability_args} had output of {output}"

            LOG.info(f"🔨 completed ability: {ability_json}")

            # add task output to memory
            self.memory.add(
                task_id=task_id,
                document=json.dumps(ability_json),
                metadatas={"role": "assistant"}
            )

            self.add_chat(task_id, "answer", json.dumps(ability_str))

            # Set the step output and is_last from AI
            step.output = answer["thoughts"]["speak"]
            step.is_last = answer["thoughts"]["last_step"]

        except json.JSONDecodeError as e:
            # Handle JSON decoding errors
            LOG.error(f"JSON error when decoding: {e}")
        except Exception as e:
            # Handle other exceptions
            LOG.error(f"Unable to generate chat response: {e}")

        # Return the completed step
        return step
