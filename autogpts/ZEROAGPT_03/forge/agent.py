import json
import pprint
import os
import shutil

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

                # check length if greater than 15 cut to last 5
                if len(self.chat_history) > 15:
                    last_msgs = self.chat_history[10:]
                    # add the first two system msgs to new_history
                    # 
                    new_history = [
                        self.chat_history[0],
                        self.chat_history[1]] + last_msgs
                    
                    self.chat_history[task_id] = new_history

        except KeyError:
            self.chat_history[task_id] = [chat_struct]
    
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

    def set_system_messages(self, task_id: str, task_input: str):
        """
        Add the call to action and response formatting
        system messages
        """
        # add system prompts to chat for task
        # set up reply json with alternative created
        system_prompt = self.prompt_engine.load_prompt("system-reformat")

        # add to messages
        # wont memory store this as static
        self.add_chat(task_id, "system", system_prompt)

        #setup call to action (cta) with task and abilities        
        ctoa_prompt_params = {
            "name": self.expert_profile["name"],
            "expertise": self.expert_profile["expertise"],
            "task": task_input,
            "abilities": self.abilities.list_abilities_for_prompt()
        }

        task_prompt = self.prompt_engine.load_prompt(
            "task-format2",
            **ctoa_prompt_params
        )

        self.add_chat(task_id, "system", task_prompt)
    
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

        try:
            role_json = profile_gen.role_find()
            LOG.info(f"role json: {role_json}")
            self.expert_profile = json.loads(role_json)
        except Exception as err:
            LOG.error(f"role JSON failed: {err}")
        
        # add system prompts to chat for task
        self.set_system_messages(task.task_id, task.input)

        return task
    
        

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
        
        # load current chat into chat completion
        try:
            chat_completion_parms = {
                "messages": self.chat_history[task_id],
                "model": os.getenv("OPENAI_MODEL")
            }

            chat_response = await chat_completion_request(
                **chat_completion_parms)
        except Exception as err:
            self.add_chat(
                    task_id,
                    "user",
                    f"[{step.step_id}] API error, please shorten your replies\error: {err}")
            step.status = "continue"
            step.is_last = False
        else:
            try:
                answer = json.loads(
                    chat_response["choices"][0]["message"]["content"])
                
                # add response to chat log
                self.add_chat(
                    task_id,
                    "assistant",
                    chat_response["choices"][0]["message"]["content"],
                )
                
                LOG.info(f"[From AI]\n{answer}")

                if "ability" in answer:

                    # Extract the ability from the answer
                    ability = answer["ability"]

                    if ability and ability["name"] != "":
                        LOG.info(f"🔨 Running Ability {ability}")

                        # Run the ability and get the output
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
                            step.status = "running"
                            step.is_last = False
                else:
                    system_prompt = self.prompt_engine.load_prompt("system-reformat")

                    # add to messages
                    # wont memory store this as static
                    self.add_chat(task_id, "system", f"There was an error with your reply\n{system_prompt}") 

                # Set the step output and is_last from AI
                step.output = answer["thoughts"]["speak"]
                
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
                    
                LOG.info(f"step status {step.status} - is_last {step.is_last}")

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
                LOG.error(f"agent.py - JSON error when decoding: {e}")
                
                step.status = "running"
                step.is_last = False

                self.add_chat(
                    task_id,
                    "user",
                    f"[{step.step_id}] Your reply was not formatted correctly. Can you please fix and try again?\error: {e}")
            except Exception as e:
                # Handle other exceptions
                LOG.error(f"execute_step error: {e}")
                LOG.info(f"chat_response: {chat_response}")
                step.status = "running"
                step.is_last = False

                self.add_chat(
                    task_id,
                    "user",
                    f"[{step.step_id}] Something went wrong with processing on our end. Please reformat your reply and try again.\error: {e}")


        if task_id in self.chat_history:
            LOG.info(f"{pprint.pformat(self.chat_history[task_id])}")
        else:
            LOG.info("No chat log yet")

        # Return the completed step
        return step
