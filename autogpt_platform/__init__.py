import glob
import json
import logging
from pathlib import Path

from .base import BaseChallenge, ChallengeInfo
from .builtin import OPTIONAL_CATEGORIES

logger = logging.getLogger(__name__)


def get_challenge_from_source_uri(source_uri: str) -> type[BaseChallenge]:
    """
    Resolve a challenge class based on the source URI prefix.
    Add new challenge providers here if needed.
    """
    from .builtin import BuiltinChallenge
    from .webarena import WebArenaChallenge

    # --- Add your new provider imports here ---
    # from .mycustom import MyCustomChallenge
    # -----------------------------------------

    provider_prefix = source_uri.split("/", 1)[0]

    # Provider registry for easy extensibility
    provider_map = {
        BuiltinChallenge.SOURCE_URI_PREFIX: BuiltinChallenge,
        WebArenaChallenge.SOURCE_URI_PREFIX: WebArenaChallenge,

    }

    provider_class = provider_map.get(provider_prefix)
    if not provider_class:
        raise ValueError(f"Cannot resolve source_uri '{source_uri}' — unknown provider prefix '{provider_prefix}'")

    return provider_class.from_source_uri(source_uri)


def get_unique_categories() -> set[str]:
    """
    Reads all challenge spec files and returns a set of all their categories.
    """
    categories = set()
    challenges_dir = Path(__file__).parent
    glob_path = f"{challenges_dir}/**/data.json"

    logger.debug(f"Scanning for challenge data files in: {challenges_dir}")

    for data_file in glob.glob(glob_path, recursive=True):
        try:
            with open(data_file, "r", encoding="utf-8") as f:
                challenge_data = json.load(f)
                file_categories = challenge_data.get("category", [])
                if isinstance(file_categories, list):
                    categories.update(file_categories)
                else:
                    logger.warning(f"Invalid category format in {data_file}")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {data_file}")
        except IOError:
            logger.error(f"Could not read file: {data_file}")

    if not categories:
        logger.warning("No categories found. Check that your data.json files are valid.")

    return categories


__all__ = [
    "BaseChallenge",
    "ChallengeInfo",
    "get_unique_categories",
    "OPTIONAL_CATEGORIES",
]
