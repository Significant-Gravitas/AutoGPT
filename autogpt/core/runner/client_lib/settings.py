from pathlib import Path

import yaml

from autogpt.core.agent import SimpleAgent


def make_user_configuration(settings_file_path: Path):
    user_configuration = SimpleAgent.build_user_configuration()

    settings_file_path.parent.mkdir(parents=True, exist_ok=True)
    print("Writing settings to", settings_file_path)
    with settings_file_path.open("w") as f:
        yaml.safe_dump(user_configuration, f)
