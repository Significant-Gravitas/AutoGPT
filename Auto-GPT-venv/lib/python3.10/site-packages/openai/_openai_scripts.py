#!/usr/bin/env python
import argparse
import logging
import sys

import openai
from openai.cli import api_register, display_error, tools_register, wandb_register

logger = logging.getLogger()
formatter = logging.Formatter("[%(asctime)s] %(message)s")
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formatter)
logger.addHandler(handler)


def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        dest="verbosity",
        default=0,
        help="Set verbosity.",
    )
    parser.add_argument("-b", "--api-base", help="What API base url to use.")
    parser.add_argument("-k", "--api-key", help="What API key to use.")
    parser.add_argument(
        "-o",
        "--organization",
        help="Which organization to run as (will use your default organization if not specified)",
    )

    def help(args):
        parser.print_help()

    parser.set_defaults(func=help)

    subparsers = parser.add_subparsers()
    sub_api = subparsers.add_parser("api", help="Direct API calls")
    sub_tools = subparsers.add_parser("tools", help="Client side tools for convenience")
    sub_wandb = subparsers.add_parser("wandb", help="Logging with Weights & Biases")

    api_register(sub_api)
    tools_register(sub_tools)
    wandb_register(sub_wandb)

    args = parser.parse_args()
    if args.verbosity == 1:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 2:
        logger.setLevel(logging.DEBUG)

    openai.debug = True
    if args.api_key is not None:
        openai.api_key = args.api_key
    if args.api_base is not None:
        openai.api_base = args.api_base
    if args.organization is not None:
        openai.organization = args.organization

    try:
        args.func(args)
    except openai.error.OpenAIError as e:
        display_error(e)
        return 1
    except KeyboardInterrupt:
        sys.stderr.write("\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
