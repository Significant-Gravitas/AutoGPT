from pathlib import Path
import logging

import yaml

from autogpt.core.agent import SimpleAgent


def make_default_settings(settings_file: str):
    user_configuration = {}
    logger = logging.getLogger("make_default_settings")
    agent_settings = SimpleAgent.compile_settings(logger, user_configuration)

    settings_file = Path(settings_file).expanduser().resolve()
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    print('Writing settings to', settings_file)
    with settings_file.open('w') as f:
        yaml.safe_dump(agent_settings.dict(), f)

