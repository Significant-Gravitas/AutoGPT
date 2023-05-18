import os, sys
import utils
import uuid
import json
import subprocess, threading
import dotenv
dotenv.load_dotenv()

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(FILE_DIR)
STATE_DIR = os.path.join(FILE_DIR, "state") 
sys.path.append(REPO_DIR)
if not os.path.exists(STATE_DIR):
    os.mkdir(STATE_DIR)
import time


def get_openai_api_key():
    return os.environ.get("OPENAI_API_KEY")


running_apis = []


def get_state(state_file):
    with open(state_file, "r") as f:
        state = json.load(f)
    return state


def set_state(state_file, state):
    with open(state_file, "w") as f:
        json.dump(state, f)


class AutoAPI:
    def __init__(self, openai_key, ai_name, ai_role, top_5_goals):
        self.openai_key = openai_key
        hex = uuid.uuid4().hex
        print(hex)
        self.state_file = os.path.join(STATE_DIR, f"state_{hex}.json")
        self.log_file = os.path.join(STATE_DIR, f"log_{hex}.json")

        newline = "\n"
        with open(os.path.join(REPO_DIR, "ai_settings.yaml"), "w") as f:
            f.write(
                f"""ai_goals:
{newline.join([f'- {goal[0]}' for goal in top_5_goals if goal[0]])}
ai_name: {ai_name}
ai_role: {ai_role}
"""
            )
        state = {
            "pending_input": None,
            "awaiting_input": False,
            "messages": [],
            "last_message_read_index": -1,
        }
        set_state(self.state_file, state)

        print(self.state_file)
        print(self.log_file)

        with open(self.log_file, "w") as f:
            subprocess.Popen(
                [
                    "python",
                    os.path.join(REPO_DIR, "ui", "api.py"),
                    openai_key,
                    self.state_file,
                ],
                cwd=REPO_DIR,
                stdout=f,
                stderr=f,
            )

    def send_message(self, message="Y"):
        state = get_state(self.state_file)
        state["pending_input"] = message
        state["awaiting_input"] = False
        set_state(self.state_file, state)

    def get_chatbot_response(self):
        while True:
            state = get_state(self.state_file)
            if (
                state["awaiting_input"]
                and state["last_message_read_index"] >= len(state["messages"]) - 1
            ):
                break
            if state["last_message_read_index"] >= len(state["messages"]) - 1:
                time.sleep(0.1)
            else:
                state["last_message_read_index"] += 1
                title, content = state["messages"][state["last_message_read_index"]]
                print(title, content)
                yield (f"**{title.strip()}** " if title else "") + utils.remove_color(
                    content
                ).replace("\n", "<br/>")
                set_state(self.state_file, state)


if __name__ == "__main__":
    print(sys.argv)
    _, openai_key, state_file = sys.argv
    os.environ["OPENAI_API_KEY"] = openai_key
    import autogpt.config.config
    from autogpt.logs import logger
    from autogpt.cli import main
    import autogpt.utils
    from autogpt.spinner import Spinner

    def add_message(title, content):
        state = get_state(state_file)
        state["messages"].append((title, content))
        set_state(state_file, state)

    def typewriter_log(title="", title_color="", content="", *args, **kwargs):
        add_message(title, content)

    def warn(message, title="", *args, **kwargs):
        add_message(title, message)

    def error(title, message="", *args, **kwargs):
        add_message(title, message)

    def clean_input(prompt=""):
        add_message(None, prompt)
        state = get_state(state_file)
        state["awaiting_input"] = True
        set_state(state_file, state)
        while state["pending_input"] is None:
            state = get_state(state_file)
            print("Waiting for input...")
            time.sleep(1)
        print("Got input")
        pending_input = state["pending_input"]
        state["pending_input"] = None
        set_state(state_file, state)
        return pending_input

    def spinner_start():
        add_message(None, "Thinking...")

    logger.typewriter_log = typewriter_log
    logger.warn = warn
    logger.error = error
    autogpt.utils.clean_input = clean_input
    # Spinner.spin = spinner_start

    sys.argv = sys.argv[:1]
    main()
