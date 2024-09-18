import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings


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


class AgentBenchmarkConfig(BaseSettings, extra="allow"):
    """
    Configuration model and loader for the AGBenchmark.

    Projects that want to use AGBenchmark should contain an agbenchmark_config folder
    with a config.json file that - at minimum - specifies the `host` at which the
    subject application exposes an Agent Protocol compliant API.
    """

    agbenchmark_config_dir: Path = Field(exclude=True)
    """Path to the agbenchmark_config folder of the subject agent application."""

    categories: list[str] | None = None
    """Categories to benchmark the agent for. If omitted, all categories are assumed."""

    host: str
    """Host (scheme://address:port) of the subject agent application."""

    reports_folder: Path = Field(None)
    """
    Path to the folder where new reports should be stored.
    Defaults to {agbenchmark_config_dir}/reports.
    """

    @classmethod
    def load(cls, config_dir: Optional[Path] = None) -> "AgentBenchmarkConfig":
        config_dir = config_dir or cls.find_config_folder()
        with (config_dir / "config.json").open("r") as f:
            return cls(
                agbenchmark_config_dir=config_dir,
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
        return self.agbenchmark_config_dir / "config.json"

    @field_validator("reports_folder", mode="before")
    def set_reports_folder(cls, value: Path, info: ValidationInfo):
        if not value:
            return info.data["agbenchmark_config_dir"] / "reports"
        return value

    def get_report_dir(self, benchmark_start_time: datetime) -> Path:
        return _calculate_info_test_path(self.reports_folder, benchmark_start_time)

    @property
    def regression_tests_file(self) -> Path:
        return self.reports_folder / "regression_tests.json"

    @property
    def success_rate_file(self) -> Path:
        return self.reports_folder / "success_rate.json"

    @property
    def challenges_already_beaten_file(self) -> Path:
        return self.agbenchmark_config_dir / "challenges_already_beaten.json"

    @property
    def temp_folder(self) -> Path:
        return self.agbenchmark_config_dir / "temp_folder"
