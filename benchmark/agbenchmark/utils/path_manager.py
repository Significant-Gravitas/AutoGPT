import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def find_agbenchmark_config(for_dir: Path) -> Path:
    """
    Find the closest ancestor folder containing an agbenchmark_config folder,
    and returns the path of that agbenchmark_config folder.

    Params:
        for_dir: The location to start searching from.

    Returns:
        Path: The applicable agbenchmark_config folder.
    """
    current_directory = for_dir
    # TODO: add support for .agbenchmark folder
    logger.debug(f"Looking for agbenchmark_config for {current_directory}")
    while current_directory != Path("/"):
        if (path := current_directory / "agbenchmark_config").exists():
            logger.debug(f"agbenchmark_config found in {current_directory}")
            if (path / "config.json").is_file():
                return path
            logger.warning(f"{path} does not contain config.json")
        current_directory = current_directory.parent
    raise FileNotFoundError(
        "No 'agbenchmark_config' directory found in the path hierarchy."
    )


class AGBenchmarkPathManager:
    def __init__(self, base_path: Path):
        if not base_path.exists():
            raise FileNotFoundError(f"base_path '{base_path}' does not exist")
        if not base_path.is_dir():
            raise ValueError(
                f"base_path must be a folder, got file path '{base_path}' instead"
            )
        self.base_path = base_path

    @classmethod
    def from_cwd(cls):
        return cls(find_agbenchmark_config(Path.cwd()))

    @property
    def config_file(self) -> Path:
        return self.base_path / "config.json"

    @property
    def challenges_already_beaten(self) -> Path:
        return self.base_path / "challenges_already_beaten.json"

    @property
    def temp_folder(self) -> Path:
        return self.base_path / "temp_folder"

    @property
    def updates_json_file(self) -> Path:
        return self.base_path / "updates.json"


PATH_MANAGER = AGBenchmarkPathManager.from_cwd()
