import json
import os
import threading
from functools import wraps
import openai
from config import Config
from file_operations import read_file, write_to_file

cfg = Config()

openai.api_key = cfg.openai_api_key

def termination_check(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Add any termination check logic here, e.g. checking for a global variable or a file
          # Replace this with your actual termination check
        result = func(*args, **kwargs)
        # open the file and read the lines
        # if threads_to_terminate.json does not exist, create it
        try:
            threads_to_terminate = read_file("threads_to_terminate.json")
            try:
                threads_to_terminate = json.loads(threads_to_terminate)
                if str(threading.get_ident()) in threads_to_terminate:
                    # remove the thread id from the dictionary
                    threads_to_terminate.pop(str(threading.get_ident()))
                    write_to_file("threads_to_terminate.json", threads_to_terminate)
                    # terminate the thread
                    print("Terminating...")
                    quit()
            except json.JSONDecodeError:
                threads_to_terminate = {}
                write_to_file("threads_to_terminate.json", threads_to_terminate)
                return result
        except FileNotFoundError:
            threads_to_terminate = {}
            write_to_file("threads_to_terminate.json", threads_to_terminate)
            return result


        return result

    return wrapper

@termination_check
# Overly simple abstraction until we create something better
def create_chat_completion(messages, model=None, temperature=int(os.environ.get("TEMPERATURE")), max_tokens=None)->str:
    """Create a chat completion using the OpenAI API"""
    if cfg.use_azure:
        response = openai.ChatCompletion.create(
            deployment_id=cfg.openai_deployment_id,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

    return response.choices[0].message["content"]
