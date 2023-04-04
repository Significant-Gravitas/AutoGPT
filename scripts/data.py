import os
from pathlib import Path


def load_prompt():
    try:
        # get directory of this file:
        file_dir = Path(os.path.dirname(os.path.realpath(__file__)))
        data_dir = file_dir / "data"
        prompt_file = data_dir / "prompt.txt"
        # Load the prompt from data/prompt.txt
        with open(prompt_file, "r") as prompt_file:
            prompt = prompt_file.read()

        # Replace {shell_type} with the current shell type 
        shell = get_shell_type()
        prompt = prompt.replace("{shell_type}", shell)

        return prompt
    except FileNotFoundError:
        print("Error: Prompt file not found", flush=True)
        return ""

def get_shell_type():
    if os.name == 'posix':
        shell = os.environ['SHELL']
        if 'zsh' in shell:
            return "zsh"
        elif 'bash' in shell:
            return "bash"
        else:
            return "unix"
    elif os.name == 'nt':
        if 'powershell' in os.environ['PATH'].lower():
            return "powershell"
        elif 'cmd' in os.environ['PATH'].lower():
            return "cmd"
        else:
            return "windows"    
    return "unknown"