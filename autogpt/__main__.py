"""Auto-GPT: A GPT powered AI Assistant"""
import logging.config

from dotenv import load_dotenv

import autogpt.cli
import autogpt.core.logging_config

if __name__ == "__main__":
    logging.config.dictConfig(autogpt.core.logging_config.logging_config)
    load_dotenv()
    autogpt.cli.main()
