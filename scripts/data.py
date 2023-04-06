import os
from pathlib import Path

SRC_DIR = Path(__file__).parent

def load_file(file_path):
    try:
        with open(SRC_DIR / file_path, "r") as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: {file_path} not found", flush=True)
        return ""

def load_prompt(init_version=False, 
                custom_message=None):
    prefix = load_prefix_prompt(init_version)
    system_msg = system_message("AutoGPT")
    postfix = load_postfix_prompt(init_version)
    custom_message = custom_message or ""
    return f"{prefix}{system_msg}{custom_message}{postfix}"

def load_prefix_prompt(init_version=False):
    file_name = "init-mode-base.txt" if init_version else "runtime-mode-base.txt"
    return load_file(f"data/{file_name}")

def load_postfix_prompt(init_version=False):
    file_name = "postfix-init-base.txt" if init_version else "postfix-runtime-base.txt"
    return load_file(f"data/{file_name}")

def system_message(system_type):
    file_name = f"{system_type}-system-message.txt"
    return load_file(f"data/{file_name}")
