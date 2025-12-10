"""
Progress Tracking Blocks for Long-Running Agents.

These blocks provide utilities for tracking progress during coding sessions,
including logging notes, code changes, test results, and git commits.

Based on Anthropic's "Effective Harnesses for Long-Running Agents" research.
"""

import logging
import uuid
from typing import Any, Optional

from backend.data.block import Block, BlockCategory, BlockOutput, BlockType
from backend.data.model import SchemaField

from .models import (
    FeatureList,
    ProgressEntry,
    ProgressEntryType,
)
from .session_manager import SessionManager

logger = logging.getLogger(__name__)


class LogProgressBlock(Block):
    """
    Log a progress entry during a coding session.

    Use this block to record important events, decisions, or milestones
    during the implementation of features.
    """

    class Input(Block.Input):
        working_directory: str = SchemaField(
            description="Directory where the project is located"
        )
        session_id: str = SchemaField(
            description="Current session ID"
        )
        entry_type: str = SchemaField(
            description="Type of progress entry (code_change, test_run, bug_fix, note, error)",
            default="note",
        )
        title: str = SchemaField(
            description="Short title for the entry"
        )
        description: str = SchemaField(
            default="",
            description="Detailed description of the progress",
        )
        feature_id: Optional[str] = SchemaField(
            default=None,
            description="Related feature ID (if applicable)",
        )
        files_changed: list[str] = SchemaField(
            default=[],
            description="List of files that were changed",
        )

    class Output(Block.Output):
        success: bool = SchemaField(
            description="Whether the entry was logged successfully"
        )
        entry_id: str = SchemaField(
            description="ID of the created entry"
        )
        error: str = SchemaField(
            description="Error message if logging failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="f6a7b8c9-d0e1-2345-f012-3456789abcde",
            description="Log a progress entry during a coding session",
            input_schema=LogProgressBlock.Input,
            output_schema=LogProgressBlock.Output,
            categories={BlockCategory.AGENT},
            block_type=BlockType.STANDARD,
        )

    async def run(
        self,
        input_data: Input,
        **kwargs,
    ) -> BlockOutput:
        try:
            manager = SessionManager(input_data.working_directory)

            # Map string to enum
            entry_type_map = {
                "code_change": ProgressEntryType.CODE_CHANGE,
                "test_run": ProgressEntryType.TEST_RUN,
                "bug_fix": ProgressEntryType.BUG_FIX,
                "note": ProgressEntryType.NOTE,
                "error": ProgressEntryType.ERROR,
                "git_commit": ProgressEntryType.GIT_COMMIT,
            }
            entry_type = entry_type_map.get(
                input_data.entry_type.lower(), ProgressEntryType.NOTE
            )

            entry_id = str(uuid.uuid4())
            entry = ProgressEntry(
                id=entry_id,
                session_id=input_data.session_id,
                entry_type=entry_type,
                title=input_data.title,
                description=input_data.description,
                feature_id=input_data.feature_id,
                files_changed=input_data.files_changed,
            )

            success = manager.add_progress_entry(entry)

            yield "success", success
            yield "entry_id", entry_id

        except Exception as e:
            logger.exception(f"Failed to log progress: {e}")
            yield "error", str(e)
            yield "success", False


class GitCommitBlock(Block):
    """
    Create a git commit with proper logging.

    This block creates a git commit and logs it to the progress file.
    Use this after making code changes to preserve the project state.
    """

    class Input(Block.Input):
        working_directory: str = SchemaField(
            description="Directory where the project is located"
        )
        session_id: str = SchemaField(
            description="Current session ID"
        )
        message: str = SchemaField(
            description="Git commit message"
        )
        files: list[str] = SchemaField(
            default=[],
            description="Specific files to commit (empty for all changes)",
        )
        feature_id: Optional[str] = SchemaField(
            default=None,
            description="Related feature ID (if applicable)",
        )

    class Output(Block.Output):
        success: bool = SchemaField(
            description="Whether the commit was successful"
        )
        commit_hash: str = SchemaField(
            description="Hash of the created commit",
            default="",
        )
        error: str = SchemaField(
            description="Error message if commit failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="a7b8c9d0-e1f2-3456-0123-456789abcdef",
            description="Create a git commit with proper progress logging",
            input_schema=GitCommitBlock.Input,
            output_schema=GitCommitBlock.Output,
            categories={BlockCategory.AGENT},
            block_type=BlockType.STANDARD,
        )

    async def run(
        self,
        input_data: Input,
        **kwargs,
    ) -> BlockOutput:
        try:
            manager = SessionManager(input_data.working_directory)

            commit_hash = manager.git_commit(
                input_data.message,
                input_data.files if input_data.files else None,
            )

            if commit_hash:
                # Log the commit
                manager.add_progress_entry(
                    ProgressEntry(
                        id=str(uuid.uuid4()),
                        session_id=input_data.session_id,
                        entry_type=ProgressEntryType.GIT_COMMIT,
                        title=f"Git commit: {input_data.message[:50]}",
                        description=input_data.message,
                        feature_id=input_data.feature_id,
                        git_commit_hash=commit_hash,
                        files_changed=input_data.files,
                    )
                )
                yield "success", True
                yield "commit_hash", commit_hash
            else:
                yield "success", False
                yield "error", "No changes to commit or commit failed"

        except Exception as e:
            logger.exception(f"Failed to create git commit: {e}")
            yield "error", str(e)
            yield "success", False


class GetProjectStatusBlock(Block):
    """
    Get the current status of a long-running project.

    This block provides a comprehensive view of the project state,
    including feature progress, recent commits, and session history.
    """

    class Input(Block.Input):
        working_directory: str = SchemaField(
            description="Directory where the project is located"
        )

    class Output(Block.Output):
        exists: bool = SchemaField(
            description="Whether a project exists at this location"
        )
        project_name: str = SchemaField(
            description="Name of the project",
            default="",
        )
        project_description: str = SchemaField(
            description="Description of the project",
            default="",
        )
        status: str = SchemaField(
            description="Current session status",
            default="",
        )
        feature_summary: dict = SchemaField(
            description="Summary of feature statuses"
        )
        is_complete: bool = SchemaField(
            description="Whether all features are passing"
        )
        session_count: int = SchemaField(
            description="Number of sessions run"
        )
        recent_progress: list[dict] = SchemaField(
            description="Recent progress entries"
        )
        recent_commits: list[dict] = SchemaField(
            description="Recent git commits"
        )
        error: str = SchemaField(
            description="Error message if status check failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="b8c9d0e1-f2a3-4567-1234-56789abcdef0",
            description="Get the current status of a long-running project",
            input_schema=GetProjectStatusBlock.Input,
            output_schema=GetProjectStatusBlock.Output,
            categories={BlockCategory.AGENT},
            block_type=BlockType.STANDARD,
        )

    async def run(
        self,
        input_data: Input,
        **kwargs,
    ) -> BlockOutput:
        try:
            manager = SessionManager(input_data.working_directory)
            status = manager.get_project_status()

            session_state = status.get("session_state")

            yield "exists", status.get("exists", False)
            yield "project_name", session_state.get("project_name", "") if session_state else ""
            yield "project_description", session_state.get("project_description", "") if session_state else ""
            yield "status", session_state.get("status", "") if session_state else ""
            yield "feature_summary", status.get("feature_summary", {})
            yield "is_complete", status.get("is_complete", False)
            yield "session_count", session_state.get("session_count", 0) if session_state else 0
            yield "recent_progress", status.get("recent_progress", [])
            yield "recent_commits", status.get("recent_commits", [])

        except Exception as e:
            logger.exception(f"Failed to get project status: {e}")
            yield "error", str(e)
            yield "exists", False


class GetFeatureListBlock(Block):
    """
    Get the complete feature list for a project.

    This block returns all features with their current statuses,
    useful for planning which features to work on.
    """

    class Input(Block.Input):
        working_directory: str = SchemaField(
            description="Directory where the project is located"
        )
        filter_status: Optional[str] = SchemaField(
            default=None,
            description="Filter by status (pending, in_progress, passing, failing, blocked, skipped)",
        )

    class Output(Block.Output):
        features: list[dict] = SchemaField(
            description="List of features"
        )
        total_count: int = SchemaField(
            description="Total number of features"
        )
        summary: dict = SchemaField(
            description="Summary of feature statuses"
        )
        error: str = SchemaField(
            description="Error message if retrieval failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="c9d0e1f2-a3b4-5678-2345-6789abcdef01",
            description="Get the complete feature list for a project",
            input_schema=GetFeatureListBlock.Input,
            output_schema=GetFeatureListBlock.Output,
            categories={BlockCategory.AGENT},
            block_type=BlockType.STANDARD,
        )

    async def run(
        self,
        input_data: Input,
        **kwargs,
    ) -> BlockOutput:
        try:
            manager = SessionManager(input_data.working_directory)
            feature_list = manager.load_feature_list()

            if not feature_list:
                yield "error", "No feature list found"
                yield "features", []
                yield "total_count", 0
                yield "summary", {}
                return

            features = feature_list.features
            if input_data.filter_status:
                features = [
                    f for f in features if f.status.value == input_data.filter_status
                ]

            feature_dicts = [
                {
                    "id": f.id,
                    "category": f.category.value,
                    "description": f.description,
                    "steps": f.steps,
                    "status": f.status.value,
                    "priority": f.priority,
                    "dependencies": f.dependencies,
                    "notes": f.notes,
                    "last_updated": f.last_updated.isoformat(),
                }
                for f in features
            ]

            yield "features", feature_dicts
            yield "total_count", len(feature_list.features)
            yield "summary", feature_list.get_progress_summary()

        except Exception as e:
            logger.exception(f"Failed to get feature list: {e}")
            yield "error", str(e)
            yield "features", []


class RevertToCommitBlock(Block):
    """
    Revert the project to a specific git commit.

    Use this block to recover from bad changes by reverting
    to a known good state.
    """

    class Input(Block.Input):
        working_directory: str = SchemaField(
            description="Directory where the project is located"
        )
        session_id: str = SchemaField(
            description="Current session ID"
        )
        commit_hash: str = SchemaField(
            description="Git commit hash to revert to"
        )
        reason: str = SchemaField(
            description="Reason for reverting"
        )

    class Output(Block.Output):
        success: bool = SchemaField(
            description="Whether the revert was successful"
        )
        error: str = SchemaField(
            description="Error message if revert failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="d0e1f2a3-b4c5-6789-3456-789abcdef012",
            description="Revert the project to a specific git commit",
            input_schema=RevertToCommitBlock.Input,
            output_schema=RevertToCommitBlock.Output,
            categories={BlockCategory.AGENT},
            block_type=BlockType.STANDARD,
        )

    async def run(
        self,
        input_data: Input,
        **kwargs,
    ) -> BlockOutput:
        try:
            manager = SessionManager(input_data.working_directory)

            success = manager.git_revert_to_commit(input_data.commit_hash)

            if success:
                # Log the revert
                manager.add_progress_entry(
                    ProgressEntry(
                        id=str(uuid.uuid4()),
                        session_id=input_data.session_id,
                        entry_type=ProgressEntryType.NOTE,
                        title=f"Reverted to commit {input_data.commit_hash[:8]}",
                        description=input_data.reason,
                        git_commit_hash=input_data.commit_hash,
                    )
                )

            yield "success", success

        except Exception as e:
            logger.exception(f"Failed to revert: {e}")
            yield "error", str(e)
            yield "success", False
