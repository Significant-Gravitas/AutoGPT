import argparse
import logging
import os
from pathlib import Path

from autogpt.commands.file_operations import ingest_file, list_files
from autogpt.config import Config
from autogpt.memory.vector import VectorMemory, get_memory
from autogpt.workspace import Workspace

cfg = Config()


def configure_logging():
    logging.basicConfig(
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(filename="log-ingestion.txt", mode="a"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("AutoGPT-Ingestion")


def ingest_directory(
    directory: str, memory: VectorMemory, workspace: Workspace, config: Config
):
    """
    Ingest all files in a directory by calling the ingest_file function for each file.

    Args:
        directory: The directory containing the files to ingest
        memory: The memory to ingest the files into
        workspace: The workspace object containing the workspace path
        config: The Config object containing the workspace path
    """
    logger = logging.getLogger("AutoGPT-Ingestion")
    try:
        files = list_files(str(workspace.get_path(directory)), config)
        for file in files:
            ingest_file(str(workspace.get_path(file)), memory)
    except Exception as e:
        logger.error(f"Error while ingesting directory '{directory}': {str(e)}")


def setup_workspace(workspace_directory, cfg) -> Workspace:
    """
    Set up the workspace directory and return the workspace object.
    Needed for Memory with file-based backend.
    Needed for data ingestion file operations.
    Args:
        workspace_directory: The workspace directory
        cfg: The Config

    Returns: The Workspace

    """
    workspace_directory = Workspace.make_workspace(workspace_directory)
    cfg.workspace_path = str(workspace_directory)
    return Workspace(workspace_directory, cfg.restrict_to_workspace)


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
        "--workspace-directory",
        type=str,
        help="The workspace directory where the data will be saved if the Memory is using a file-based backend.",
    )
    args = parser.parse_args()

    workspace = setup_workspace(args.workspace_directory, cfg)

    # Initialize memory
    memory = get_memory(cfg)
    logger.debug("Using memory of type: " + memory.__class__.__name__)

    if args.file:
        try:
            ingest_file(str(workspace.get_path(args.file)), memory)
            logger.info(f"File '{args.file}' ingested successfully.")
        except Exception as e:
            logger.error(f"Error while ingesting file '{args.file}': {str(e)}")
    elif args.dir:
        try:
            ingest_directory(args.dir, memory, workspace, cfg)
            logger.info(f"Directory '{args.dir}' ingested successfully.")
        except Exception as e:
            logger.error(f"Error while ingesting directory '{args.dir}': {str(e)}")
    else:
        logger.warn(
            "Please provide either a file path (--file) or a directory name (--dir)"
            " inside the auto_gpt_workspace directory as input."
        )


if __name__ == "__main__":
    main()
