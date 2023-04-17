import os, sys

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(FILE_DIR)
sys.path.append(REPO_DIR)
import autogpt.config.config

config = autogpt.config.config.Config()

def get_openai_api_key():
    return config.openai_api_key

def setup(openai_key, llm_mode, ai_name, ai_role, top_5_goals):
    config.openai_api_key = openai_key

def send_message(message="Y"):
    return

def get_chatbot_response():
    import time
    for _ in range(4):
        time.sleep(1)
        yield "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
