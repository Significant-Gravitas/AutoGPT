import json
import pprint
import os
import shutil
from datetime import datetime

from pathlib import Path

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

from forge.sdk.ai_planning import AIPlanning

import openai

# from elevenlabs import generate, play

LOG = ForgeLogger(__name__)


class ForgeAgent(Agent):
    def __init__(self, database: AgentDB, workspace: Workspace):
        super().__init__(database, workspace)
        
        # initialize chat history per uuid
        self.chat_history = {}

        # memory storage
        self.memory = None

        # expert profile
        self.expert_profile = None

        # setup chatcompletion to achieve this task
        # with custom prompt that will generate steps
        self.prompt_engine = PromptEngine(os.getenv("OPENAI_MODEL"))

        # ai plan
        self.ai_plan = None

        # instruction messages
        self.instruction_msgs = {}

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        # set instruction amount to 0
        self.instruct_amt = 0

        # create task
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
        
        # initalize memstore for task
        try:
            # initialize memstore
            cwd = self.workspace.get_cwd_path(task.task_id)
            chroma_dir = f"{cwd}/chromadb"
            os.makedirs(chroma_dir)
            self.memory = ChromaMemStore(chroma_dir+"/")
            LOG.info(f"🧠 Created memorystore @ {os.getenv('AGENT_WORKSPACE')}/{task.task_id}/chroma")
        except Exception as err:
            LOG.error(f"memstore creation failed: {err}")
        
        # get role for task
        # use AI to get the proper role experts for the task
        profile_gen = ProfileGenerator(
            task=task,
            prompt_engine=self.prompt_engine
        )

        self.expert_profile = await profile_gen.role_find()
        
        # add system prompts to chat for task
        self.instruction_msgs[task.task_id] = []
        await self.set_instruction_messages(task.task_id, task.input)

        return task
    
    def add_chat(self, 
        task_id: str, 
        role: str, 
        content: str,
        is_function: bool = False,
        function_name: str = None):
        
        if is_function:
            chat_struct = {
                "role": role,
                "name": function_name,
                "content": content
            }
        else:
            chat_struct = {
                "role": role, 
                "content": content
            }
        
        try:
            # cut down on messages being repeated in chat
            if chat_struct not in self.chat_history[task_id]:
                self.chat_history[task_id].append(chat_struct)

        except KeyError:
            self.chat_history[task_id] = [chat_struct]

    async def cut_chat(self, task_id, task_input):
        """
        Cut down chat and remake it with last 4 messages,
        including beginning needed system messages
        Also rebuild the steps and professional
        """
        # clear chat and rebuild
        self.chat_history[task_id] = []

        # get last two most recent messages if not system messages
        last_chat_history = []
        print(f"len of chat history @ {task_id}: {len(self.chat_history[task_id])}")
        print(f"len of instruction msgs @ {task_id}: {len(self.instruction_msgs[task_id])}")

        if len(self.chat_history[task_id]) > len(self.instruction_msgs[task_id]):
            # get past two assistant messages to add context
            last_chat_history.append(self.chat_history[task_id][-2])
            print(last_chat_history)

        for imsg in self.instruction_msgs[task_id]:
            msg = {
                "role": imsg[0],
                "content": imsg[1]
            }

            self.chat_history[task_id].append(msg)

        self.chat_history[task_id] += last_chat_history

    async def set_instruction_messages(self, task_id: str, task_input: str):
        """
        Add the call to action and response formatting
        system and user messages
        """
        # sys format and abilities as system way
        #---------------------------------------------------
        # # add system prompts to chat for task
        # # set up reply json with alternative created
        # system_prompt = self.prompt_engine.load_prompt("system-reformat")

        # # add to messages
        # # wont memory store this as static
        # self.add_chat(task_id, "system", system_prompt)

        # # add abilities prompt
        # abilities_prompt = self.prompt_engine.load_prompt(
        #     "abilities-list",
        #     **{"abilities": self.abilities.list_abilities_for_prompt()}
        # )

        # self.add_chat(task_id, "system", abilities_prompt)

        # ----------------------------------------------------
        # AI planning and steps way

        # add system prompts to chat for task
        # set up reply json with alternative created
        system_prompt = self.prompt_engine.load_prompt("system-reformat")

        # add to messages
        # wont memory store this as static
        LOG.info(f"🖥️  {system_prompt}")
        self.instruction_msgs[task_id].append(("system", system_prompt))
        self.add_chat(task_id, "system", system_prompt)

        # add abilities prompt
        abilities_prompt = self.prompt_engine.load_prompt(
            "abilities-list",
            **{"abilities": self.abilities.list_abilities_for_prompt()}
        )

        LOG.info(f"🖥️  {abilities_prompt}")
        self.instruction_msgs[task_id].append(("system", abilities_prompt))
        self.add_chat(task_id, "system", abilities_prompt)

        # add role system prompt
        try:
            role_prompt_params = {
                "name": self.expert_profile["name"],
                "expertise": self.expert_profile["expertise"]
            }
        except Exception as err:
            LOG.error(f"""
                Error generating role, using default\n
                Name: Joe Anybody\n
                Expertise: Project Manager\n
                err: {err}""")
            role_prompt_params = {
                "name": "Joe Anybody",
                "expertise": "Project Manager"
            }
            
        role_prompt = self.prompt_engine.load_prompt(
            "role-statement",
            **role_prompt_params
        )

        LOG.info(f"🖥️  {role_prompt}")
        self.instruction_msgs[task_id].append(("system", role_prompt))
        self.add_chat(task_id, "system", role_prompt)

        # setup call to action (cta) with task and abilities
        # use ai to plan the steps
        self.ai_plan = AIPlanning(
            task_input,
            self.abilities.list_abilities_for_prompt()
        )

        plan_steps_prompt = await self.ai_plan.create_steps()
        LOG.info(f"🖥️ planned steps\n{plan_steps_prompt}")
        plan_steps_prompt = json.loads(plan_steps_prompt)

        ctoa_prompt_params = {
            "steps": plan_steps_prompt["steps"]
        }

        task_prompt = self.prompt_engine.load_prompt(
            "step-work",
            **ctoa_prompt_params
        )

        LOG.info(f"🤓 {task_prompt}")
        self.instruction_msgs[task_id].append(("user", task_prompt))
        self.add_chat(task_id, "user", task_prompt)
        # ----------------------------------------------------

    def copy_to_temp(self, task_id: str):
        """
        Copy files created from cwd to temp
        """
        cwd = self.workspace.get_cwd_path(task_id)
        tmp = self.workspace.get_temp_path(task_id)

        for filename in os.listdir(cwd):
            if ".sqlite3" not in filename:
                file_path = os.path.join(cwd, filename)
                if os.path.isfile(file_path):
                    LOG.info(f"copying {str(file_path)} to {tmp}")
                    shutil.copy(file_path, tmp)

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        task = await self.db.get_task(task_id)

        # Create a new step in the database
        # have AI determine last step
        step = await self.db.create_step(
            task_id=task_id,
            input=step_request,
            additional_input=step_request.additional_input,
            is_last=False
        )

        step.status = "running"

        # used in some chat messages
        timestamp = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        
        # load current chat into chat completion
        try:
            chat_completion_parms = {
                "messages": self.chat_history[task_id],
                "model": os.getenv("OPENAI_MODEL"),
                "temperature": 1
            }

            chat_response = await chat_completion_request(
                **chat_completion_parms)
        except Exception as err:
            LOG.error("API token error. Cut down messages.")
            await self.cut_chat(task_id, task.input)
            
            step.status = "completed"
        else:
            # add response to chat log
            self.add_chat(
                task_id,
                "assistant",
                chat_response["choices"][0]["message"]["content"],
            )

            try:
                answer = json.loads(
                    chat_response["choices"][0]["message"]["content"])
                
                # make sure about reply format
                if "ability" not in answer or "thoughts" not in answer:
                    system_prompt = self.prompt_engine.load_prompt("system-reformat")
                    self.add_chat(
                        task_id,
                        "system",
                        f"[{timestamp}] Your reply was not in the given JSON format.\n{system_prompt}")
                    step.status = "completed"
                else:
                    if "ability" in answer:

                        # Extract the ability from the answer
                        ability = answer["ability"]

                        if ability and (
                            ability["name"] != "" or
                            ability["name"] == None or
                            ability["name"] == "None"):
                            LOG.info(f"🔨 Running Ability {ability}")

                            # Run the ability and get the output
                            try:
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

                                    LOG.info(f"🔨 Ability Output\n{output}")
                            except Exception as err:
                                LOG.error(f"Ability run failed: {err}")
                                self.add_chat(
                                    task_id=task_id,
                                    role="system",
                                    content=f"[{timestamp}] Ability {ability['name']} failed to run: {err}"
                                )

                            # change output to string if there is output
                            if isinstance(output, bytes):
                                output = output.decode()
                            
                            # add to converstion
                            # add arguments to function content, if any
                            if "args" in ability:
                                ccontent = f"[Arguments {ability['args']}]: {output} "
                            else:
                                ccontent = output

                            self.add_chat(
                                task_id=task_id,
                                role="function",
                                content=ccontent,
                                is_function=True,
                                function_name=ability["name"]
                            )

                            if ability["name"] == "finish":
                                step.status = "completed"
                                step.is_last = True
                                self.copy_to_temp(task_id)
                            else:
                                step.status = "completed"

                    # Set the step output and is_last from AI
                    step.output = answer["thoughts"]["speak"]
                    LOG.info(f"🤖 Thoughts")
                    LOG.info(f"🤖 {answer['thoughts']}")

                    # if "last_step" in answer["thoughts"]:
                    #     last_step_code = answer["thoughts"]["last_step"]
                    #     if isinstance(last_step_code, str):
                    #         try:
                    #             last_step_code = int(last_step_code)
                    #         except:
                    #             last_step_code = 0

                    #     if (bool(last_step_code)):
                    #         step.status = "completed"
                    #         step.is_last = True
                    #         self.copy_to_temp(task_id)
                    #     else:
                    #         step.status = "running"
                    #         step.is_last = False
                        
                    LOG.info(f"⏳ step status {step.status} is_last? {step.is_last}")

                    # have ai speak through speakers
                    # cant use yet due to pydantic version differences
                    # audio = generate(
                    #     text=answer["thoughts"]["speak"],
                    #     voice="Dorothy",
                    #     model="eleven_multilingual_v2"
                    # )

                    # play(audio)

            except json.JSONDecodeError as e:
                # Handle JSON decoding errors
                LOG.error(f"agent.py - JSON error when decoding chat_response: {e}")
                LOG.error(f"{chat_response}")
                step.status = "completed"
                step.is_last = False
                
                system_prompt = self.prompt_engine.load_prompt("system-reformat")
                self.add_chat(
                    task_id,
                    "system",
                    f"[{timestamp}] Your reply was not JSON formatted.\n{system_prompt}")
            except Exception as e:
                # Handle other exceptions
                LOG.error(f"execute_step error: {e}")
                LOG.info(f"chat_response: {chat_response}")
                step.status = "completed"
                step.is_last = False

                self.add_chat(
                    task_id,
                    "system",
                    f"[{timestamp}] Something went wrong with processing on our end. Please reformat your reply and try again.")

        # dump whole chat log at last step
        if step.is_last and task_id in self.chat_history:
            LOG.info(f"{pprint.pformat(self.chat_history[task_id])}")

        # Return the completed step
        return step
