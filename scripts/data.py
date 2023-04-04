import os
from pathlib import Path


# def load_prompt():
#     try:
#         # get directory of this file:
#         file_dir = Path(os.path.dirname(os.path.realpath(__file__)))
#         data_dir = file_dir / "data"
#         prompt_file = data_dir / "prompt.txt"
#         # Load the promt from data/prompt.txt
#         with open(prompt_file, "r") as prompt_file:
#             prompt = prompt_file.read()
#
#         return prompt
#     except FileNotFoundError:
#         print("Error: Prompt file not found", flush=True)
#         return ""

def load_prompt():
    try:
        # Get the directory of this file:
        file_dir = Path(os.path.dirname(os.path.realpath(__file__)))
        data_dir = file_dir / "data"
        prompt_file = data_dir / "prompt.txt"
        # Load the prompt from data/prompt.txt
        with open(prompt_file, "r") as prompt_file:
            prompt = prompt_file.read()

        # Load prompt commands from prompt_commands.txt
        prompt_commands_file = data_dir / "prompt_commands.txt"
        with open(prompt_commands_file, "r") as commands_file:
            commands = commands_file.read()

        # Replace <<AI COMMANDS>> with the content of prompt_commands.txt
        prompt = prompt.replace("<<AI COMMANDS>>", commands)
        print(prompt)
        return prompt
    except FileNotFoundError:
        print("Error: Prompt file not found", flush=True)
        return ""


load_prompt()
