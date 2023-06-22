"""Auto-GPT: A GPT powered AI Assistant"""
import logging.config

import autogpt.cli
import autogpt.core.logging_config

if __name__ == "__main__":
    logging.config.dictConfig(autogpt.core.logging_config.logging_config)
    autogpt.cli.main()
