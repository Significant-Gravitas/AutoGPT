import logging
import shutil
from datetime import datetime
from pathlib import Path

from autogpt.core.workspace.simple import SimpleWorkspace


class TimestampedWorkspace(SimpleWorkspace):
    """A workspace which creates a timestamped directory on each run."""

    run_prefix = ("run-",)

    @staticmethod
    def setup_workspace(settings: "AgentSettings", logger: logging.Logger) -> Path:
        workspace_root = SimpleWorkspace.setup_workspace(settings, logger)

        run_root = workspace_root / TimestampedWorkspace.run_folder_prefix + str(
            datetime.now()
        )
        run_root.mkdir(parents=True, exist_ok=True)

        settings.workspace.configuration.root = str(run_root)
        settings.workspace.configuration.parent = str(workspace_root)

        shutil.copy(
            workspace_root / "agent_settings.json", run_root / "agent_settings.json"
        )

        run_log_path = run_root / "logs"
        run_log_path.mkdir(parents=True, exist_ok=True)
        (run_log_path / "debug.log").touch()
        (run_log_path / "cycle.log").touch()
        return run_root
