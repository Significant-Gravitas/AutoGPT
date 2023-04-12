import argparse
import logging
from config import Config
from memory import get_memory
from file_operations import ingest_file, ingest_directory

cfg = Config()

def configure_logging():
    logging.basicConfig(filename='log-ingestion.txt',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
    return logging.getLogger('AutoGPT-Ingestion')


def main():
    logger = configure_logging()

    parser = argparse.ArgumentParser(description="Ingest a file or a directory with multiple files into memory. Make sure to set your .env before running this script.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="The file to ingest.")
    group.add_argument("--dir", type=str, help="The directory containing the files to ingest.")
    parser.add_argument("--init", action='store_true', help="Init the memory and wipe its content", default=False)
    args = parser.parse_args()


    # Initialize memory
    memory = get_memory(cfg, init=args.init)
    print('Using memory of type: ' + memory.__class__.__name__)

    if args.file:
        try:
            ingest_file(args.file, memory)
            print(f"File '{args.file}' ingested successfully.")
        except Exception as e:
            logger.error(f"Error while ingesting file '{args.file}': {str(e)}")
            print(f"Error while ingesting file '{args.file}': {str(e)}")
    elif args.dir:
        try:
            ingest_directory(args.dir, memory)
            print(f"Directory '{args.dir}' ingested successfully.")
        except Exception as e:
            logger.error(f"Error while ingesting directory '{args.dir}': {str(e)}")
            print(f"Error while ingesting directory '{args.dir}': {str(e)}")
    else:
        print("Please provide either a file path (--file) or a directory name (--dir) inside the auto_gpt_workspace directory as input.")


if __name__ == "__main__":
    main()
