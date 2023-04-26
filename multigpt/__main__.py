"""Main script for the autogpt package."""
import logging
from autogpt.agent.agent import Agent
from autogpt.args import parse_arguments
from autogpt.config import check_openai_api_key
from autogpt.logs import logger
from multigpt.multi_config import MultiConfig
from multigpt.multi_agent_manager import MultiAgentManager
from multigpt.setup import prompt_user


def main() -> None:
    cfg = MultiConfig()
    check_openai_api_key()
    parse_arguments()
    logger.set_level(logging.DEBUG if cfg.debug_mode else logging.INFO)

    multi_agent_manager = MultiAgentManager(cfg)

    prompt_user(cfg, multi_agent_manager)

    multi_agent_manager.start_interaction_loop()


if __name__ == "__main__":
    main()
