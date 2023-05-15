import logging
from pprint import pprint

from autogpt.core.agent import SimpleAgent


def make_default_settings():
    user_configuration = {}
    logger = logging.getLogger("make_default_settings")
    agent_settings = SimpleAgent.compile_settings(logger, user_configuration)
    pprint(agent_settings.dict())
