import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

HELICONE_GRAPHQL_LOGS = os.getenv("HELICONE_GRAPHQL_LOGS", "").lower() == "true"


def get_list_of_file_paths(
    challenge_dir_path: str | Path, artifact_folder_name: str
) -> list[Path]:
    source_dir = Path(challenge_dir_path) / artifact_folder_name
    if not source_dir.exists():
        return []
    return list(source_dir.iterdir())


def copy_challenge_artifacts_into_workspace(
    challenge_dir_path: str | Path, artifact_folder_name: str, workspace: str | Path
) -> None:
    file_paths = get_list_of_file_paths(challenge_dir_path, artifact_folder_name)
    for file_path in file_paths:
        if file_path.is_file():
            shutil.copy(file_path, workspace)
