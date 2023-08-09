"""
TEMPORARY FILE FOR TESTING PURPOSES ONLY WILL BE REMOVED SOON!
-------------------------------------------------------------
PLEASE IGNORE
-------------------------------------------------------------
"""

import os
import typing
from pathlib import Path

import dotenv

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
) -> typing.Union[typing.Dict[str, typing.Any], Exception]:
    """Generate a response to a list of messages using OpenAI's API"""
    try:
        return openai.ChatCompletion.create(
            model=model,
            messages=messages,
            user="TheForge",
        )
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        exit()


def run(task: str) -> None:
    """Runs the agent for benchmarking"""
    print("Running agent")
    steps = plan(task)
    execute_plan(steps)


def execute_plan(plan: typing.List[str]) -> None:
    """Each step is valid python, join the steps together into a python script and execute it"""
    script = "\n".join(plan)
    print(f"Executing script: \n{script}")
    exec(script)


def plan(task: str) -> typing.List[str]:
    """Returns a list of tasks that needs to be executed to complete the task"""
    abilities = """
    plan(task: str) -> typing.List[str]
    write_file(contents: str, filepath: str) -> bool
    read_file(filepath:str) -> typing.Optional[str]
    append_to_file(contents: str, filepath: str, to_start: bool) -> bool
    read_webpage(url: str) -> str
    """
    json_format = """
        {
        "steps": [
            "write_file('The capital is xxx', 'answer.txt')",
            "read_file('file_to_read.txt')",
        ]
    }
    """
    planning_prompt = f"""Answer in json format:
    Determine the steps needed to complete the following task using only the defined list of steps with the parameters provided:
    {task}
    ---
    Possible steps
    {abilities}

    ---
    Example answer:
    {json_format}
    """
    response = chat_completion_request(
        messages=[{"role": "user", "content": planning_prompt}]
    )

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
        print(f"Unable to write file: {e}")
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
        print(f"Unable to read file: {e}")
    return contents


def read_webpage(url: str) -> typing.Optional[str]:
    """Checks if the url is valid then reads the contents of the webpage"""
    contents = None
    try:
        response = requests.get(url)
        if response.status_code == 200:
            contents = response.text
    except Exception as e:
        print(f"Unable to read webpage: {e}")
    return contents
