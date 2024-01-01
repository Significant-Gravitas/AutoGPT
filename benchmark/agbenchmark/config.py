import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, TypedDict

from pydantic import BaseModel


def _calculate_info_test_path(base_path: Path, benchmark_start_time: datetime) -> Path:
    """
    Calculates the path to the directory where the test report will be saved.
    """
    # Ensure the reports path exists
    base_path.mkdir(parents=True, exist_ok=True)

    # Get current UTC date-time stamp
    date_stamp = benchmark_start_time.strftime("%Y%m%dT%H%M%S")

    # Default run name
    run_name = "full_run"

    # Map command-line arguments to their respective labels
    arg_labels = {
        "--test": None,
        "--category": None,
        "--maintain": "maintain",
        "--improve": "improve",
        "--explore": "explore",
    }

    # Identify the relevant command-line argument
    for arg, label in arg_labels.items():
        if arg in sys.argv:
            test_arg = sys.argv[sys.argv.index(arg) + 1] if label is None else None
            run_name = arg.strip("--")
            if test_arg:
                run_name = f"{run_name}_{test_arg}"
            break

    # Create the full new directory path with ISO standard UTC date-time stamp
    report_path = base_path / f"{date_stamp}_{run_name}"

    # Ensure the new directory is created
    # FIXME: this is not a desirable side-effect of loading the config
    report_path.mkdir(exist_ok=True)

    return report_path


class WorkspaceConfig(TypedDict):
    input: str
    output: str


class AgentBenchmarkConfig(BaseModel):
    """
    Configuration model and loader for the AGBenchmark.

    Attributes:
    - agbenchmark_config_path: The path to the agbenchmark config that this object was created from.
    - reports_folder: The path to the folder where the benchmark reports will be stored.
    - host: The host where the benchmark is run.
    """

    agbenchmark_config_path: Path
    workspace: WorkspaceConfig | None = None
    host: str

    @classmethod
    def load(cls, base_path: Optional[Path] = None) -> "AgentBenchmarkConfig":
        base_path = base_path or cls.find_config_folder()
        with (base_path / "config.json").open("r") as f:
            return cls(
                agbenchmark_config_path=base_path,
                **json.load(f),
            )

    @staticmethod
    def find_config_folder(for_dir: Path = Path.cwd()) -> Path:
        """
        Find the closest ancestor folder containing an agbenchmark_config folder,
        and returns the path of that agbenchmark_config folder.
        """
        current_directory = for_dir
        while current_directory != Path("/"):
            if (path := current_directory / "agbenchmark_config").exists():
                if (path / "config.json").is_file():
                    return path
            current_directory = current_directory.parent
        raise FileNotFoundError(
            "No 'agbenchmark_config' directory found in the path hierarchy."
        )

    @property
    def config_file(self) -> Path:
        return self.agbenchmark_config_path / "config.json"

    @property
    def agent_home_directory(self) -> Path:
        return Path(self.agbenchmark_config_path).resolve().parent

    @property
    def reports_folder(self) -> Path:
        return self.agbenchmark_config_path / "reports"

    def get_reports_path(self, benchmark_start_time: datetime) -> Path:
        return _calculate_info_test_path(self.reports_folder, benchmark_start_time)

    @property
    def regression_reports_path(self) -> Path:
        return self.reports_folder / "regression_tests.json"

    @property
    def success_rate_path(self) -> Path:
        return self.reports_folder / "success_rate.json"

    @property
    def challenges_already_beaten(self) -> Path:
        return self.agbenchmark_config_path / "challenges_already_beaten.json"

    @property
    def updates_json_file(self) -> Path:
        return self.agbenchmark_config_path / "updates.json"

    @property
    def temp_folder(self) -> Path:
        return self.agbenchmark_config_path / "temp_folder"
