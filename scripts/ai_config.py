import shutil
import time
from pathlib import Path

import data
import yaml

# Constants
SRC_DIR = Path(__file__).parent
PROMPT_START_FILE = SRC_DIR / "data" / "prompts" / "prompt_start.txt"
SAVE_FILE = SRC_DIR / "data" / "ai_settings" / "last_settings.yaml"
HISTORY_DIR = SRC_DIR / "data" / "ai_settings" / "history"


def get_timestamped_filename(prefix, extension):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


class AIConfig:
    def __init__(self, ai_name="", ai_role="", ai_goals=[]):
        self.ai_name = ai_name
        self.ai_role = ai_role
        self.ai_goals = ai_goals

    @classmethod
    def load(cls, config_file=SAVE_FILE):
        # Load variables from yaml file if it exists
        try:
            with open(config_file) as file:
                config_params = yaml.load(file, Loader=yaml.FullLoader)
        except FileNotFoundError:
            config_params = {}

        ai_name = config_params.get("ai_name", "")
        ai_role = config_params.get("ai_role", "")
        ai_goals = config_params.get("ai_goals", [])

        return cls(ai_name, ai_role, ai_goals)

    @classmethod
    def load_from_history(cls, filename, history_path=HISTORY_DIR):
        try:
            with open(history_path / filename) as file:
                config_params = yaml.load(file, Loader=yaml.FullLoader)
        except FileNotFoundError:
            print(f"File {filename} not found in the history folder.")
            return None

        ai_name = config_params.get("ai_name", "")
        ai_role = config_params.get("ai_role", "")
        ai_goals = config_params.get("ai_goals", [])

        return cls(ai_name, ai_role, ai_goals)

    @classmethod
    def get_history_files(cls, history_path=HISTORY_DIR):
        return [f.name for f in history_path.glob("*.yaml")]

    def save(self, config_file=SAVE_FILE):
        config = {"ai_name": self.ai_name,
                  "ai_role": self.ai_role, "ai_goals": self.ai_goals}
        with open(config_file, "w") as file:
            yaml.dump(config, file)

    def save_to_history(self):
        timestamped_filename = get_timestamped_filename("settings", "yaml")
        shutil.copy2(SAVE_FILE, HISTORY_DIR / timestamped_filename)

    def construct_full_prompt(self):
        # Load prompt_start from the text file
        with open(PROMPT_START_FILE, "r") as file:
            prompt_start = file.read()

        # Construct full prompt
        full_prompt = f"You are {self.ai_name}, {self.ai_role}\n{prompt_start}\n\nGOALS:\n\n"
        for i, goal in enumerate(self.ai_goals):
            full_prompt += f"{i+1}. {goal}\n"

        full_prompt += f"\n\n{data.load_prompt()}"
        return full_prompt
