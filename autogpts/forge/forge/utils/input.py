import logging

import click

from forge.config.config import Config

logger = logging.getLogger(__name__)


def clean_input(config: "Config", prompt: str = ""):
    try:
        # ask for input, default when just pressing Enter is y
        logger.debug("Asking user via keyboard...")

        return click.prompt(
            text=prompt, prompt_suffix=" ", default="", show_default=False
        )
    except KeyboardInterrupt:
        logger.info("You interrupted AutoGPT")
        logger.info("Quitting...")
        exit(0)
