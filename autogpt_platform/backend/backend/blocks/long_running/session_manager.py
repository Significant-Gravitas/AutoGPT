"""
Session manager for long-running agent framework.

This module handles:
- Creating and managing session state files
- Reading and writing feature lists
- Managing progress logs
- Git operations for version control
"""

import json
import logging
import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import (
    FeatureCategory,
    FeatureList,
    FeatureListItem,
    FeatureStatus,
    LongRunningSessionState,
    ProgressEntry,
    ProgressEntryType,
    ProgressLog,
    SessionStatus,
)

logger = logging.getLogger(__name__)

# Default file names
SESSION_STATE_FILE = ".autogpt_session_state.json"
FEATURE_LIST_FILE = "feature_list.json"
PROGRESS_LOG_FILE = "autogpt_progress.txt"
INIT_SCRIPT_FILE = "init.sh"


class SessionManager:
    """
    Manages long-running agent session state, features, and progress.

    This class provides file-based persistence for agent sessions,
    following the patterns from Anthropic's long-running agent research.
    """

    def __init__(self, working_directory: str):
        """
        Initialize the session manager.

        Args:
            working_directory: The directory where session files will be stored
        """
        self.working_directory = Path(working_directory)
        self.working_directory.mkdir(parents=True, exist_ok=True)

        self._session_state: Optional[LongRunningSessionState] = None
        self._feature_list: Optional[FeatureList] = None
        self._progress_log: Optional[ProgressLog] = None

    @property
    def session_state_path(self) -> Path:
        return self.working_directory / SESSION_STATE_FILE

    @property
    def feature_list_path(self) -> Path:
        return self.working_directory / FEATURE_LIST_FILE

    @property
    def progress_log_path(self) -> Path:
        return self.working_directory / PROGRESS_LOG_FILE

    @property
    def init_script_path(self) -> Path:
        return self.working_directory / INIT_SCRIPT_FILE

    # === Session State Management ===

    def load_session_state(self) -> Optional[LongRunningSessionState]:
        """Load session state from file if it exists."""
        if self.session_state_path.exists():
            try:
                with open(self.session_state_path) as f:
                    data = json.load(f)
                self._session_state = LongRunningSessionState.model_validate(data)
                return self._session_state
            except Exception as e:
                logger.error(f"Failed to load session state: {e}")
                return None
        return None

    def save_session_state(self, state: LongRunningSessionState) -> bool:
        """Save session state to file."""
        try:
            state.last_updated = datetime.utcnow()
            with open(self.session_state_path, "w") as f:
                json.dump(state.model_dump(mode="json"), f, indent=2, default=str)
            self._session_state = state
            return True
        except Exception as e:
            logger.error(f"Failed to save session state: {e}")
            return False

    def create_session_state(
        self,
        project_name: str,
        project_description: str,
    ) -> LongRunningSessionState:
        """Create a new session state for a project."""
        state = LongRunningSessionState(
            id=str(uuid.uuid4()),
            project_name=project_name,
            project_description=project_description,
            status=SessionStatus.INITIALIZING,
            working_directory=str(self.working_directory),
            feature_list_path=str(self.feature_list_path),
            progress_log_path=str(self.progress_log_path),
            init_script_path=str(self.init_script_path),
        )
        self.save_session_state(state)
        return state

    def get_or_create_session_state(
        self, project_name: str, project_description: str
    ) -> LongRunningSessionState:
        """Get existing session state or create a new one."""
        existing = self.load_session_state()
        if existing:
            return existing
        return self.create_session_state(project_name, project_description)

    def update_session_status(
        self, status: SessionStatus, session_id: Optional[str] = None
    ) -> bool:
        """Update the session status."""
        state = self.load_session_state()
        if not state:
            return False

        state.status = status
        if session_id:
            state.current_session_id = session_id
        if status == SessionStatus.WORKING:
            state.session_count += 1

        return self.save_session_state(state)

    # === Feature List Management ===

    def load_feature_list(self) -> Optional[FeatureList]:
        """Load feature list from file if it exists."""
        if self.feature_list_path.exists():
            try:
                with open(self.feature_list_path) as f:
                    data = json.load(f)
                self._feature_list = FeatureList.model_validate(data)
                return self._feature_list
            except Exception as e:
                logger.error(f"Failed to load feature list: {e}")
                return None
        return None

    def save_feature_list(self, feature_list: FeatureList) -> bool:
        """Save feature list to file."""
        try:
            with open(self.feature_list_path, "w") as f:
                json.dump(feature_list.model_dump(mode="json"), f, indent=2, default=str)
            self._feature_list = feature_list
            return True
        except Exception as e:
            logger.error(f"Failed to save feature list: {e}")
            return False

    def create_feature_list(
        self,
        project_name: str,
        project_description: str,
        features: list[dict],
    ) -> FeatureList:
        """
        Create a new feature list for a project.

        Args:
            project_name: Name of the project
            project_description: Description of the project
            features: List of feature dictionaries with at minimum 'description'
        """
        feature_items = []
        for i, f in enumerate(features):
            item = FeatureListItem(
                id=f.get("id", f"feature_{i+1:03d}"),
                category=FeatureCategory(
                    f.get("category", FeatureCategory.FUNCTIONAL.value)
                ),
                description=f["description"],
                steps=f.get("steps", []),
                status=FeatureStatus.PENDING,
                priority=f.get("priority", 5),
                dependencies=f.get("dependencies", []),
            )
            feature_items.append(item)

        feature_list = FeatureList(
            project_name=project_name,
            project_description=project_description,
            features=feature_items,
        )
        self.save_feature_list(feature_list)
        return feature_list

    def update_feature_status(
        self,
        feature_id: str,
        status: FeatureStatus,
        session_id: str,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Update the status of a feature.

        IMPORTANT: This is the ONLY way features should be modified.
        Never remove or edit feature descriptions.
        """
        feature_list = self.load_feature_list()
        if not feature_list:
            return False

        feature = feature_list.get_feature_by_id(feature_id)
        if not feature:
            logger.error(f"Feature {feature_id} not found")
            return False

        feature.status = status
        feature.last_updated = datetime.utcnow()
        feature.updated_by_session = session_id
        if notes:
            feature.notes = notes

        return self.save_feature_list(feature_list)

    def get_next_feature_to_work_on(self) -> Optional[FeatureListItem]:
        """Get the next feature to work on based on priority and status."""
        feature_list = self.load_feature_list()
        if not feature_list:
            return None
        return feature_list.get_next_feature()

    # === Progress Log Management ===

    def load_progress_log(self) -> ProgressLog:
        """Load progress log from file or create a new one."""
        if self.progress_log_path.exists():
            try:
                # Parse the text-based progress log
                entries = self._parse_progress_log_text()
                state = self.load_session_state()
                project_name = state.project_name if state else "Unknown Project"
                self._progress_log = ProgressLog(
                    project_name=project_name, entries=entries
                )
                return self._progress_log
            except Exception as e:
                logger.error(f"Failed to load progress log: {e}")

        # Return empty log if file doesn't exist or parsing failed
        state = self.load_session_state()
        project_name = state.project_name if state else "Unknown Project"
        return ProgressLog(project_name=project_name, entries=[])

    def _parse_progress_log_text(self) -> list[ProgressEntry]:
        """Parse a text-based progress log into structured entries."""
        entries: list[ProgressEntry] = []

        if not self.progress_log_path.exists():
            return entries

        content = self.progress_log_path.read_text()
        current_entry: dict = {}
        current_session: str = "unknown"

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Parse session headers
            if line.startswith("=== Session:"):
                current_session = line.replace("=== Session:", "").strip().rstrip("=")
                continue

            # Parse timestamp and entry
            if line.startswith("[") and "]" in line:
                # Save previous entry if exists
                if current_entry:
                    entries.append(ProgressEntry(**current_entry))

                # Parse new entry
                try:
                    timestamp_str, rest = line.split("]", 1)
                    timestamp_str = timestamp_str.strip("[")
                    timestamp = datetime.fromisoformat(timestamp_str)
                except (ValueError, IndexError):
                    timestamp = datetime.utcnow()
                    rest = line

                current_entry = {
                    "id": str(uuid.uuid4()),
                    "session_id": current_session,
                    "entry_type": ProgressEntryType.NOTE,
                    "timestamp": timestamp,
                    "title": rest.strip()[:100],
                    "description": rest.strip(),
                }
            elif current_entry:
                # Append to current entry's description
                current_entry["description"] += "\n" + line

        # Save last entry
        if current_entry:
            entries.append(ProgressEntry(**current_entry))

        return entries

    def add_progress_entry(self, entry: ProgressEntry) -> bool:
        """Add a new entry to the progress log."""
        try:
            # Append to text file in human-readable format
            with open(self.progress_log_path, "a") as f:
                timestamp = entry.timestamp.isoformat()
                f.write(f"\n[{timestamp}] {entry.entry_type.value.upper()}: {entry.title}\n")
                if entry.description and entry.description != entry.title:
                    f.write(f"  {entry.description}\n")
                if entry.feature_id:
                    f.write(f"  Feature: {entry.feature_id}\n")
                if entry.git_commit_hash:
                    f.write(f"  Commit: {entry.git_commit_hash}\n")
                if entry.files_changed:
                    f.write(f"  Files: {', '.join(entry.files_changed)}\n")

            return True
        except Exception as e:
            logger.error(f"Failed to add progress entry: {e}")
            return False

    def start_session_log(self, session_id: str) -> bool:
        """Add a session start entry to the progress log."""
        try:
            with open(self.progress_log_path, "a") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"=== Session: {session_id} ===\n")
                f.write(f"=== Started: {datetime.utcnow().isoformat()} ===\n")
                f.write(f"{'='*60}\n")

            entry = ProgressEntry(
                id=str(uuid.uuid4()),
                session_id=session_id,
                entry_type=ProgressEntryType.SESSION_START,
                title=f"Session {session_id} started",
            )
            return self.add_progress_entry(entry)
        except Exception as e:
            logger.error(f"Failed to start session log: {e}")
            return False

    def end_session_log(self, session_id: str, summary: str) -> bool:
        """Add a session end entry to the progress log."""
        try:
            entry = ProgressEntry(
                id=str(uuid.uuid4()),
                session_id=session_id,
                entry_type=ProgressEntryType.SESSION_END,
                title=f"Session {session_id} ended",
                description=summary,
            )
            self.add_progress_entry(entry)

            with open(self.progress_log_path, "a") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"=== Session {session_id} Summary ===\n")
                f.write(f"{summary}\n")
                f.write(f"{'='*60}\n\n")

            return True
        except Exception as e:
            logger.error(f"Failed to end session log: {e}")
            return False

    # === Git Operations ===

    def initialize_git(self) -> bool:
        """Initialize a git repository in the working directory."""
        try:
            # Check if already initialized
            git_dir = self.working_directory / ".git"
            if git_dir.exists():
                logger.info("Git repository already initialized")
                return True

            # Initialize git
            result = subprocess.run(
                ["git", "init"],
                cwd=self.working_directory,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error(f"Git init failed: {result.stderr}")
                return False

            # Create .gitignore
            gitignore_path = self.working_directory / ".gitignore"
            gitignore_content = """# AutoGPT Long-Running Agent Files
.autogpt_session_state.json
*.log
*.pyc
__pycache__/
.env
.venv/
node_modules/
.DS_Store
"""
            gitignore_path.write_text(gitignore_content)

            # Initial commit
            subprocess.run(
                ["git", "add", "."],
                cwd=self.working_directory,
                capture_output=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "Initial commit - AutoGPT long-running agent setup"],
                cwd=self.working_directory,
                capture_output=True,
            )

            # Update session state
            state = self.load_session_state()
            if state:
                state.git_repo_initialized = True
                self.save_session_state(state)

            return True
        except Exception as e:
            logger.error(f"Failed to initialize git: {e}")
            return False

    def git_commit(self, message: str, files: Optional[list[str]] = None) -> Optional[str]:
        """
        Create a git commit with the given message.

        Args:
            message: Commit message
            files: Optional list of specific files to commit (defaults to all changes)

        Returns:
            The commit hash if successful, None otherwise
        """
        try:
            # Add files
            if files:
                for f in files:
                    subprocess.run(
                        ["git", "add", f],
                        cwd=self.working_directory,
                        capture_output=True,
                    )
            else:
                subprocess.run(
                    ["git", "add", "-A"],
                    cwd=self.working_directory,
                    capture_output=True,
                )

            # Check if there are changes to commit
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.working_directory,
                capture_output=True,
                text=True,
            )
            if not status.stdout.strip():
                logger.info("No changes to commit")
                return None

            # Commit
            result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.working_directory,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error(f"Git commit failed: {result.stderr}")
                return None

            # Get commit hash
            hash_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.working_directory,
                capture_output=True,
                text=True,
            )
            return hash_result.stdout.strip() if hash_result.returncode == 0 else None
        except Exception as e:
            logger.error(f"Failed to create git commit: {e}")
            return None

    def get_git_log(self, limit: int = 10) -> list[dict]:
        """Get recent git commits."""
        try:
            result = subprocess.run(
                [
                    "git",
                    "log",
                    f"-{limit}",
                    "--pretty=format:%H|%s|%ai|%an",
                ],
                cwd=self.working_directory,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return []

            commits = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 4:
                    commits.append({
                        "hash": parts[0],
                        "message": parts[1],
                        "date": parts[2],
                        "author": parts[3],
                    })
            return commits
        except Exception as e:
            logger.error(f"Failed to get git log: {e}")
            return []

    def git_revert_to_commit(self, commit_hash: str) -> bool:
        """Revert to a specific commit."""
        try:
            result = subprocess.run(
                ["git", "reset", "--hard", commit_hash],
                cwd=self.working_directory,
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to revert to commit: {e}")
            return False

    # === Init Script Management ===

    def create_init_script(
        self,
        commands: list[str],
        description: str = "",
    ) -> bool:
        """
        Create an init.sh script for environment setup.

        Args:
            commands: List of shell commands to include
            description: Description of what the script does
        """
        try:
            script_content = f"""#!/bin/bash
# AutoGPT Long-Running Agent Init Script
# {description}
# Generated: {datetime.utcnow().isoformat()}

set -e  # Exit on error

echo "=== AutoGPT Environment Setup ==="

# Check if we're in the right directory
if [ ! -f "{FEATURE_LIST_FILE}" ]; then
    echo "Error: {FEATURE_LIST_FILE} not found. Are you in the project directory?"
    exit 1
fi

"""
            for cmd in commands:
                script_content += f"echo \"Running: {cmd}\"\n"
                script_content += f"{cmd}\n\n"

            script_content += """
echo "=== Environment setup complete ==="
"""

            self.init_script_path.write_text(script_content)

            # Make executable
            os.chmod(self.init_script_path, 0o755)

            return True
        except Exception as e:
            logger.error(f"Failed to create init script: {e}")
            return False

    def run_init_script(self) -> tuple[bool, str]:
        """
        Run the init.sh script.

        Returns:
            Tuple of (success, output)
        """
        if not self.init_script_path.exists():
            return False, "Init script not found"

        try:
            result = subprocess.run(
                ["bash", str(self.init_script_path)],
                cwd=self.working_directory,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            output = result.stdout + result.stderr
            return result.returncode == 0, output
        except subprocess.TimeoutExpired:
            return False, "Init script timed out"
        except Exception as e:
            return False, str(e)

    # === Status Queries ===

    def get_project_status(self) -> dict:
        """Get a comprehensive status of the project."""
        state = self.load_session_state()
        feature_list = self.load_feature_list()
        progress_log = self.load_progress_log()
        git_log = self.get_git_log(5)

        status = {
            "exists": state is not None,
            "session_state": state.model_dump() if state else None,
            "feature_summary": (
                feature_list.get_progress_summary() if feature_list else None
            ),
            "is_complete": feature_list.is_complete() if feature_list else False,
            "recent_progress": [
                e.model_dump() for e in progress_log.get_recent_entries(5)
            ],
            "recent_commits": git_log,
        }
        return status

    def is_initialized(self) -> bool:
        """Check if the project has been initialized."""
        state = self.load_session_state()
        return state is not None and state.status != SessionStatus.INITIALIZING
