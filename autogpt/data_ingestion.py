import argparse
import logging

from autogpt.config import Config
from autogpt.commands.file_operations import ingest_file, search_files
from autogpt.memory import get_memory

cfg = Config()


def configure_logging():
    logging.basicConfig(
        filename="log-ingestion.txt",
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
    )
    return logging.getLogger("AutoGPT-Ingestion")


def ingest_directory(directory, memory, args):
    """
    Ingest all files in a directory by calling the ingest_file function for each file.

    :param directory: The directory containing the files to ingest
    :param memory: An object with an add() method to store the chunks in memory
    """
    try:
        files = search_files(directory)
        for file in files:
            ingest_file(file, memory, args.max_length, args.overlap)
    except Exception as e:
        print(f"Error while ingesting directory '{directory}': {str(e)}")


def main() -> None:
    logger = configure_logging()

    parser = argparse.ArgumentParser(
        description="Ingest a file or a directory with multiple files into memory. "
        "Make sure to set your .env before running this script."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="The file to ingest.")
    group.add_argument(
        "--dir", type=str, help="The directory containing the files to ingest."
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Init the memory and wipe its content (default: False)",
        default=False,
    )
    parser.add_argument(
        "--overlap",
        type=int,
        help="The overlap size between chunks when ingesting files (default: 200)",
        default=200,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        help="The max_length of each chunk when ingesting files (default: 4000)",
        default=4000,
    )

    args = parser.parse_args()

    # Initialize memory
    memory = get_memory(cfg, init=args.init)
    print("Using memory of type: " + memory.__class__.__name__)

    if args.file:
        try:
            ingest_file(args.file, memory, args.max_length, args.overlap)
            print(f"File '{args.file}' ingested successfully.")
        except Exception as e:
            logger.error(f"Error while ingesting file '{args.file}': {str(e)}")
            print(f"Error while ingesting file '{args.file}': {str(e)}")
    elif args.dir:
        try:
            ingest_directory(args.dir, memory, args)
            print(f"Directory '{args.dir}' ingested successfully.")
        except Exception as e:
            logger.error(f"Error while ingesting directory '{args.dir}': {str(e)}")
            print(f"Error while ingesting directory '{args.dir}': {str(e)}")
    else:
        print(
            "Please provide either a file path (--file) or a directory name (--dir)"
            " inside the auto_gpt_workspace directory as input."
        )


if __name__ == "__main__":
    main()
