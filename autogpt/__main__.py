"""Auto-GPT: A GPT powered AI Assistant"""
import autogpt.core.cli
import autogpt.core.config
import logging.config

if __name__ == "__main__":
    logging.config.dictConfig(autogpt.core.config.logging_config)
    autogpt.core.cli.main()
