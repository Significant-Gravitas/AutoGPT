import glob
import json
import logging
from pathlib import Path

from .base import BaseChallenge, ChallengeInfo
from .builtin import OPTIONAL_CATEGORIES

logger = logging.getLogger(__name__)


def get_challenge_from_source_uri(source_uri: str) -> type[BaseChallenge]:
    from .builtin import BuiltinChallenge
    from .webarena import WebArenaChallenge

    provider_prefix = source_uri.split("/", 1)[0]

    if provider_prefix == BuiltinChallenge.SOURCE_URI_PREFIX:
        return BuiltinChallenge.from_source_uri(source_uri)

    if provider_prefix == WebArenaChallenge.SOURCE_URI_PREFIX:
        return WebArenaChallenge.from_source_uri(source_uri)

    raise ValueError(f"Cannot resolve source_uri '{source_uri}'")


def get_unique_categories() -> set[str]:
    """
    Reads all challenge spec files and returns a set of all their categories.
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


__all__ = [
    "BaseChallenge",
    "ChallengeInfo",
    "get_unique_categories",
    "OPTIONAL_CATEGORIES",
]
