"""Auto-GPT: A GPT powered AI Assistant"""
import autogpt.cli
import autogpt.core.logging_config
import logging.config

if __name__ == "__main__":
    logging.config.dictConfig(autogpt.core.logging_config.logging_config)
    autogpt.cli.main()
