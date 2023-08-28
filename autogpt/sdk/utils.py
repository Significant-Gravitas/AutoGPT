"""
TEMPORARY FILE FOR TESTING PURPOSES ONLY WILL BE REMOVED SOON!
-------------------------------------------------------------
PLEASE IGNORE
-------------------------------------------------------------
"""

import glob
import os
import typing
from pathlib import Path

import dotenv

from .forge_log import CustomLogger

LOG = CustomLogger(__name__)

dotenv.load_dotenv()

import openai
import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential

PROJECT_DIR = Path().resolve()
workspace = os.path.join(PROJECT_DIR, "agbenchmark/workspace")


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(
    messages: typing.List[typing.Dict[str, str]],
    functions: typing.List[typing.Dict[str, str]] | None = None,
    function_call: typing.Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0,
) -> typing.Union[typing.Dict[str, typing.Any], Exception]:
    """Generate a response to a list of messages using OpenAI's API"""
    try:
        return openai.ChatCompletion.create(
            model=model,
            messages=messages,
            user="TheForge",
            temperature=temperature,
        )
    except Exception as e:
        LOG.info("Unable to generate ChatCompletion response")
        LOG.info(f"Exception: {e}")
        exit()


def run(task: str):
    """Runs the agent for benchmarking"""
    LOG.info("Running agent")
    steps = plan(task)
    execute_plan(steps)
    # check for artifacts in workspace
    items = glob.glob(os.path.join(workspace, "*"))
    if items:
        artifacts = []
        LOG.info(f"Found {len(items)} artifacts in workspace")
        for item in items:
            with open(item, "r") as f:
                item_contents = f.read()
            path_within_workspace = os.path.relpath(item, workspace)
            artifacts.append(
                {
                    "file_name": os.path.basename(item),
                    "uri": f"file://{path_within_workspace}",
                    "contents": item_contents,
                }
            )
    return artifacts


def execute_plan(plan: typing.List[str]) -> None:
    """Each step is valid python, join the steps together into a python script and execute it"""
    script = "\n".join(plan)
    LOG.info(f"Executing script: \n{script}")
    exec(script)


def plan(task: str) -> typing.List[str]:
    """Returns a list of tasks that needs to be executed to complete the task"""
    abilities = """
    write_file(contents='The content you want to write', filepath='file_to_write.txt')
    read_file(filepath='file_to_write.txt')
    """
    json_format = """
        {
        "steps": [
            "write_file(contents='The capital is xxx', filepath='answer.txt')",
            "read_file(filepath='file_to_read.txt')",
        ]
    }
    """
    planning_prompt = f"""Answer in json format:
    Determine the steps needed to complete the following task :
    {task}
    ---
    Possible steps:
    {abilities}

    ---
    Example answer:
    {json_format}

    ---
    As you can see, we only use hard coded values when calling the functions.
    Please write your answer below:
    """
    messages = [{"role": "user", "content": planning_prompt}]

    response = chat_completion_request(messages=messages)

    import json

    plan = json.loads(response.choices[0].message.content)
    return plan["steps"]


def append_to_file(contents: str, filepath: str, to_start: bool) -> bool:
    """Reads in a file then writes the file out with the contents appended to the end or start"""
    if workspace not in filepath:
        filepath = os.path.join(workspace, filepath)
    file_contents = read_file(filepath)
    if file_contents is None:
        file_contents = ""
    if to_start:
        contents += file_contents
    else:
        contents = file_contents + contents
    return write_file(contents, filepath)


def write_file(contents: str, filepath: str) -> bool:
    """Creates directory for the file if it doesn't exist, then writes the file"""
    if workspace not in filepath:
        filepath = os.path.join(workspace, filepath)
    success = False
    directory = os.path.dirname(filepath)
    os.makedirs(directory, exist_ok=True)
    try:
        with open(filepath, "w") as f:
            f.write(contents)
        success = True
    except Exception as e:
        LOG.info(f"Unable to write file: {e}")
    return success


def read_file(filepath: str) -> typing.Optional[str]:
    """Reads in the contents of a file"""
    if workspace not in filepath:
        filepath = os.path.join(workspace, filepath)
    contents = None
    try:
        with open(filepath, "r") as f:
            contents = f.read()
    except Exception as e:
        LOG.info(f"Unable to read file: {e}")
    return contents


def read_webpage(url: str) -> typing.Optional[str]:
    """Checks if the url is valid then reads the contents of the webpage"""
    contents = None
    try:
        response = requests.get(url)
        if response.status_code == 200:
            contents = response.text
    except Exception as e:
        LOG.info(f"Unable to read webpage: {e}")
    return contents


if __name__ == "__main__":
    test_messages = [{"role": "user", "content": "Hello, how are you?"}]
    response = chat_completion_request(test_messages)
    LOG.info(response)
