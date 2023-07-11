# radio charts, logs, helper functions for tests, anything else relevant.
import glob
from pathlib import Path


def calculate_info_test_path(benchmarks_folder_path: Path) -> str:
    INFO_TESTS_PATH = benchmarks_folder_path / "reports"

    if not INFO_TESTS_PATH.exists():
        INFO_TESTS_PATH.mkdir(parents=True, exist_ok=True)
        return str(INFO_TESTS_PATH / "1.json")
    else:
        json_files = glob.glob(str(INFO_TESTS_PATH / "*.json"))
        file_count = len(json_files)
        run_name = f"{file_count + 1}.json"
        new_file_path = INFO_TESTS_PATH / run_name
        return str(new_file_path)
