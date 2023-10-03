from codecs import ignore_errors
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
)

LOG = ForgeLogger(__name__)

import openai

import pandas as pd
import csv
import traceback
import requests
from bs4 import BeautifulSoup


class ChatGPT():

    temperature = 0
    max_tokens = 2000
    top_p = 1
    frequency_penalty = 0
    presence_penalty = 0.6
    model = "gpt-3.5-turbo" # "gpt-4"


    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise Exception("Please set OPENAI_API_KEY environment variable."
            "You can obtain API key from https://platform.openai.com/account/api-keys")
        openai.api_key = api_key        

    @property
    def _default_params(self):
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "model": self.model,
        }

    def chat(self, prompt):

        params = {
            **self._default_params,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }
        response = openai.ChatCompletion.create(**params)
        LOG.info(json.dumps(response, indent=4))
        return response["choices"][0]["message"]["content"]

class Executor:

    def run(self, code, globals_env=None, locals_env=None):
        try:
            tmp_code = ""
            for line in code.split("\n"):
                if not line.startswith("```"):
                    tmp_code += line + "\n"

            exec(tmp_code, globals_env, locals_env)
        except Exception as e:
            return str(traceback.format_exc())[-100:]
        return None


prompt_create_file = """Your task is:

{}

INSTRUCTIONS

Use python code to solve the task. You can use pandas package if needed. Load data with tabular separator.

Please start the file with `# new_file filename.py`. 

Please call the function that is solving the task if needed.

Please return python code only.
""" 

prompt_run_python = """Your task is:

{}

Please return python code only that solves the task.
""" 


class ForgeAgent(Agent):
    def __init__(self, database: AgentDB, workspace: Workspace):
        super().__init__(database, workspace)
        

    
    async def create_task(self, task_request: TaskRequestBody) -> Task:
        try:
            task = await self.db.create_task(
                input=task_request.input,
                additional_input=task_request.additional_input
            )

            LOG.info(
                f"📦 Task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
            )
            self.chat = ChatGPT()
            self.steps = [] 

        except Exception as err:
            LOG.error(f"create_task failed: {err}")
            raise err

        return task

    def files_to_prompt(self, wd, files):
        prompt = ""
        for f in files:
            fpath = os.path.join(wd, f)
            content = ""
            with open(fpath, "r") as fin:
                content = fin.read()
            lines = content.split("\n")
            if len(lines) == 0:
                prompt += f"\n - file {f} is empty\n\n"
            elif len(lines) < 15:
                prompt += f"\n - file {f} includes\n\n"
                prompt += "```\n"
                prompt += content + "```\n\n"
            else:
                prompt += f"\n - file {f} has {len(lines)} lines, below are first 15 lines from it\n\n```"
                for l in lines[:10]:
                    prompt += f"{l}\n"
                prompt += "```\n\n"
        return prompt


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

        self.steps += [step]

        LOG.info(f"Steps :{len(self.steps)}")

        wd = os.path.join(str(self.workspace.base_path), task_id)
        LOG.info(wd)
        if not os.path.exists(wd):
            os.makedirs(wd)
        os.chdir(wd)

        files = [f for f in os.listdir(wd) if os.path.isfile(os.path.join(wd, f))]
        LOG.info(str(files))

        inc_files = self.files_to_prompt(wd, files)

        task_prompt = task.input
        if inc_files != "":
            task_prompt += f"\n\nThere are {len(files)} files in the current directory. Files are listed below:\n\n"
            task_prompt += inc_files

        prompt = prompt_create_file.format(task_prompt)

        

        LOG.info(f"Prompt: {prompt}")
        response = self.chat.chat(prompt)
        LOG.info(f"Response: {response}")
    
        code = response
        tmp_code = ""
        code_fname = ""
        for line in code.split("\n"):
            if line.startswith("# new_file"):
                code_fname = line[10:]
            if not line.startswith("```"):
                tmp_code += line + "\n"

        LOG.info(f"Create file {code_fname}")
        if code_fname != "":
            with open(code_fname, "w") as fin:
                fin.write(tmp_code)

        executor = Executor()
        error = executor.run(tmp_code, globals(), locals())
        if error is not None:
            LOG.info("ERROR")
            LOG.info(error)
            # try to rescue
            prompt += "\nChatGPT returned code:\n```python\n"
            prompt += tmp_code + "\n```\n"
            prompt += "\nThe error is\n"
            prompt += error
            prompt += "\n\nPlease fix the code. Do NOT return the same code. Return python code only."

            LOG.info(f"Prompt: {prompt}")
            response = self.chat.chat(prompt)
            LOG.info(f"Response: {response}")
    
            code = response
            tmp_code = ""
            code_fname = ""
            for line in code.split("\n"):
                if line.startswith("# new_file"):
                    code_fname = line[10:]
                if not line.startswith("```"):
                    tmp_code += line + "\n"

            LOG.info(f"Create file {code_fname}")
            if code_fname != "":
                with open(code_fname, "w") as fin:
                    fin.write(tmp_code)

            executor = Executor()
            error = executor.run(tmp_code, globals(), locals())
            if error is not None:
                LOG.info("Second ERROR")
                LOG.info(error)


        LOG.info(os.getcwd())
        for file_path in os.listdir(os.getcwd()):
            LOG.info(file_path)
            await self.db.create_artifact(
                task_id=task_id,
                step_id=step.step_id,
                file_name=file_path,
                relative_path="",
                agent_created=True,
            )

        if len(self.steps) >= 1:
            step.is_last = True

        # Return the completed step
        return step