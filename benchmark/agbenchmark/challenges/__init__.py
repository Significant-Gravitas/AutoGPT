import glob
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_unique_categories() -> set[str]:
    """
    Find all data.json files in the directory relative to this file and its
    subdirectories, read the "category" field from each file, and return a set of unique
    categories.
    """
    categories = set()

    challenges_dir = Path(__file__).parent
    glob_path = f"{challenges_dir}/**/data.json"

    for data_file in glob.glob(glob_path, recursive=True):
        with open(data_file, "r") as f:
            try:
                challenge_data = json.load(f)
                categories.update(challenge_data.get("category", []))
            except json.JSONDecodeError:
                logger.error(f"Error: {data_file} is not a valid JSON file.")
                continue
            except IOError:
                logger.error(f"IOError: file could not be read: {data_file}")
                continue

    return categories
