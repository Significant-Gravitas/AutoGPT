"""
Long-Running Coding Block.

This block is used for subsequent sessions after initialization.
It follows the pattern of:
1. Getting oriented (reading progress, git logs, feature list)
2. Choosing a feature to work on
3. Making incremental progress
4. Testing the feature
5. Committing changes and updating progress

Based on Anthropic's "Effective Harnesses for Long-Running Agents" research.
"""

import logging
import uuid
from typing import Any, Optional

from backend.data.block import Block, BlockCategory, BlockOutput, BlockType
from backend.data.model import SchemaField

from .models import (
    FeatureStatus,
    ProgressEntry,
    ProgressEntryType,
    SessionStatus,
)
from .session_manager import SessionManager

logger = logging.getLogger(__name__)


class LongRunningCodingBlock(Block):
    """
    Start a coding session for a long-running agent project.

    This block is used AFTER initialization to work on features incrementally.
    It provides context about the project state and the next feature to work on.

    The coding agent workflow:
    1. Read the feature list and progress log
    2. Choose the highest-priority incomplete feature
    3. Implement the feature
    4. Test the feature thoroughly
    5. Mark the feature as complete and commit changes
    """

    class Input(Block.Input):
        working_directory: str = SchemaField(
            description="Directory where the project is located"
        )
        verify_basic_functionality: bool = SchemaField(
            default=True,
            description="Run basic verification before starting new work",
        )
        max_context_entries: int = SchemaField(
            default=20,
            description="Maximum number of progress entries to include in context",
        )

    class Output(Block.Output):
        session_id: str = SchemaField(
            description="Unique identifier for this session"
        )
        project_name: str = SchemaField(
            description="Name of the project"
        )
        project_description: str = SchemaField(
            description="Description of the project"
        )
        current_feature: dict = SchemaField(
            description="The feature to work on in this session"
        )
        feature_summary: dict = SchemaField(
            description="Summary of feature statuses (passing, failing, pending, etc.)"
        )
        recent_progress: list[dict] = SchemaField(
            description="Recent progress entries for context"
        )
        recent_commits: list[dict] = SchemaField(
            description="Recent git commits for context"
        )
        init_script_output: str = SchemaField(
            description="Output from running the init script (if verify_basic_functionality is True)",
            default="",
        )
        is_project_complete: bool = SchemaField(
            description="Whether all features are passing"
        )
        status: str = SchemaField(
            description="Status of session initialization"
        )
        error: str = SchemaField(
            description="Error message if session start failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="b2c3d4e5-f6a7-8901-bcde-f12345678901",
            description="Start a coding session for a long-running agent project, providing context and the next feature to work on",
            input_schema=LongRunningCodingBlock.Input,
            output_schema=LongRunningCodingBlock.Output,
            categories={BlockCategory.AGENT},
            block_type=BlockType.AGENT,
        )

    async def run(
        self,
        input_data: Input,
        **kwargs,
    ) -> BlockOutput:
        session_id = str(uuid.uuid4())[:8]

        try:
            # Initialize session manager
            manager = SessionManager(input_data.working_directory)

            # Load session state
            state = manager.load_session_state()
            if not state:
                yield "error", f"No session state found at {input_data.working_directory}. Run initializer first."
                yield "status", "failed"
                return

            if state.status == SessionStatus.INITIALIZING:
                yield "error", "Project initialization not complete. Run initializer first."
                yield "status", "failed"
                return

            # Update session status
            manager.update_session_status(SessionStatus.WORKING, session_id)

            # Start session log
            manager.start_session_log(session_id)

            # Run verification if requested
            init_output = ""
            if input_data.verify_basic_functionality:
                success, output = manager.run_init_script()
                init_output = output
                if not success:
                    logger.warning(f"Init script failed: {output}")
                    manager.add_progress_entry(
                        ProgressEntry(
                            id=str(uuid.uuid4()),
                            session_id=session_id,
                            entry_type=ProgressEntryType.ERROR,
                            title="Init script failed",
                            description=output[:500],
                        )
                    )

            # Load feature list
            feature_list = manager.load_feature_list()
            if not feature_list:
                yield "error", "No feature list found. Run initializer first."
                yield "status", "failed"
                return

            # Get next feature to work on
            next_feature = feature_list.get_next_feature()
            current_feature_dict: dict[str, Any] = {}

            if next_feature:
                # Mark feature as in progress
                manager.update_feature_status(
                    next_feature.id,
                    FeatureStatus.IN_PROGRESS,
                    session_id,
                )

                current_feature_dict = {
                    "id": next_feature.id,
                    "category": next_feature.category.value,
                    "description": next_feature.description,
                    "steps": next_feature.steps,
                    "priority": next_feature.priority,
                    "dependencies": next_feature.dependencies,
                    "notes": next_feature.notes,
                }

                # Log feature start
                manager.add_progress_entry(
                    ProgressEntry(
                        id=str(uuid.uuid4()),
                        session_id=session_id,
                        entry_type=ProgressEntryType.FEATURE_START,
                        title=f"Started working on: {next_feature.description[:50]}",
                        feature_id=next_feature.id,
                    )
                )

            # Get context
            feature_summary = feature_list.get_progress_summary()
            progress_log = manager.load_progress_log()
            recent_progress = [
                e.model_dump(mode="json")
                for e in progress_log.get_recent_entries(input_data.max_context_entries)
            ]
            recent_commits = manager.get_git_log(10)

            # Check if complete
            is_complete = feature_list.is_complete()

            yield "session_id", session_id
            yield "project_name", state.project_name
            yield "project_description", state.project_description
            yield "current_feature", current_feature_dict
            yield "feature_summary", feature_summary
            yield "recent_progress", recent_progress
            yield "recent_commits", recent_commits
            yield "init_script_output", init_output
            yield "is_project_complete", is_complete
            yield "status", "success" if not is_complete else "complete"

        except Exception as e:
            logger.exception(f"Failed to start coding session: {e}")
            yield "error", str(e)
            yield "status", "failed"


class FeatureCompleteBlock(Block):
    """
    Mark a feature as complete and commit changes.

    This block should be used after successfully implementing and testing a feature.
    It updates the feature status, creates a git commit, and logs progress.
    """

    class Input(Block.Input):
        working_directory: str = SchemaField(
            description="Directory where the project is located"
        )
        session_id: str = SchemaField(
            description="Current session ID"
        )
        feature_id: str = SchemaField(
            description="ID of the feature to mark as complete"
        )
        commit_message: str = SchemaField(
            description="Git commit message describing the changes"
        )
        test_results: str = SchemaField(
            default="",
            description="Description of how the feature was tested",
        )
        files_changed: list[str] = SchemaField(
            default=[],
            description="List of files that were modified",
        )

    class Output(Block.Output):
        success: bool = SchemaField(
            description="Whether the feature was marked complete successfully"
        )
        commit_hash: str = SchemaField(
            description="Git commit hash if commit was successful",
            default="",
        )
        next_feature: dict = SchemaField(
            description="The next feature to work on (if any)"
        )
        remaining_features: int = SchemaField(
            description="Number of features still pending"
        )
        error: str = SchemaField(
            description="Error message if operation failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="c3d4e5f6-a7b8-9012-cdef-123456789012",
            description="Mark a feature as complete, commit changes, and get the next feature to work on",
            input_schema=FeatureCompleteBlock.Input,
            output_schema=FeatureCompleteBlock.Output,
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

            # Update feature status
            success = manager.update_feature_status(
                input_data.feature_id,
                FeatureStatus.PASSING,
                input_data.session_id,
                notes=f"Tested: {input_data.test_results}" if input_data.test_results else None,
            )

            if not success:
                yield "error", f"Failed to update feature status for {input_data.feature_id}"
                yield "success", False
                return

            # Create git commit
            commit_hash = manager.git_commit(
                input_data.commit_message,
                input_data.files_changed if input_data.files_changed else None,
            )

            # Log progress
            manager.add_progress_entry(
                ProgressEntry(
                    id=str(uuid.uuid4()),
                    session_id=input_data.session_id,
                    entry_type=ProgressEntryType.FEATURE_COMPLETE,
                    title=f"Completed feature: {input_data.feature_id}",
                    description=input_data.test_results,
                    feature_id=input_data.feature_id,
                    git_commit_hash=commit_hash,
                    files_changed=input_data.files_changed,
                )
            )

            # Get next feature
            feature_list = manager.load_feature_list()
            next_feature = feature_list.get_next_feature() if feature_list else None
            next_feature_dict: dict[str, Any] = {}

            if next_feature:
                next_feature_dict = {
                    "id": next_feature.id,
                    "category": next_feature.category.value,
                    "description": next_feature.description,
                    "steps": next_feature.steps,
                    "priority": next_feature.priority,
                }

            # Count remaining
            remaining = 0
            if feature_list:
                summary = feature_list.get_progress_summary()
                remaining = summary.get("pending", 0) + summary.get("failing", 0)

            yield "success", True
            yield "commit_hash", commit_hash or ""
            yield "next_feature", next_feature_dict
            yield "remaining_features", remaining

        except Exception as e:
            logger.exception(f"Failed to complete feature: {e}")
            yield "error", str(e)
            yield "success", False


class FeatureFailedBlock(Block):
    """
    Mark a feature as failed and log the issue.

    Use this block when a feature cannot be completed due to blockers,
    bugs, or other issues. It preserves the work done and notes for the next session.
    """

    class Input(Block.Input):
        working_directory: str = SchemaField(
            description="Directory where the project is located"
        )
        session_id: str = SchemaField(
            description="Current session ID"
        )
        feature_id: str = SchemaField(
            description="ID of the feature that failed"
        )
        failure_reason: str = SchemaField(
            description="Description of why the feature failed"
        )
        commit_partial_work: bool = SchemaField(
            default=True,
            description="Whether to commit any partial work done",
        )
        commit_message: str = SchemaField(
            default="",
            description="Git commit message if committing partial work",
        )

    class Output(Block.Output):
        success: bool = SchemaField(
            description="Whether the failure was logged successfully"
        )
        commit_hash: str = SchemaField(
            description="Git commit hash if partial work was committed",
            default="",
        )
        error: str = SchemaField(
            description="Error message if operation failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="d4e5f6a7-b8c9-0123-def0-123456789abc",
            description="Mark a feature as failed, log the issue, and optionally commit partial work",
            input_schema=FeatureFailedBlock.Input,
            output_schema=FeatureFailedBlock.Output,
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

            # Update feature status
            success = manager.update_feature_status(
                input_data.feature_id,
                FeatureStatus.FAILING,
                input_data.session_id,
                notes=f"Failed: {input_data.failure_reason}",
            )

            if not success:
                yield "error", f"Failed to update feature status for {input_data.feature_id}"
                yield "success", False
                return

            # Commit partial work if requested
            commit_hash = ""
            if input_data.commit_partial_work and input_data.commit_message:
                commit_hash = manager.git_commit(input_data.commit_message) or ""

            # Log progress
            manager.add_progress_entry(
                ProgressEntry(
                    id=str(uuid.uuid4()),
                    session_id=input_data.session_id,
                    entry_type=ProgressEntryType.FEATURE_FAILED,
                    title=f"Feature failed: {input_data.feature_id}",
                    description=input_data.failure_reason,
                    feature_id=input_data.feature_id,
                    git_commit_hash=commit_hash if commit_hash else None,
                )
            )

            yield "success", True
            yield "commit_hash", commit_hash

        except Exception as e:
            logger.exception(f"Failed to mark feature as failed: {e}")
            yield "error", str(e)
            yield "success", False


class SessionEndBlock(Block):
    """
    End a coding session properly with a summary.

    This block should be called at the end of every coding session to:
    - Write a session summary to the progress log
    - Commit any uncommitted changes
    - Update the session state
    """

    class Input(Block.Input):
        working_directory: str = SchemaField(
            description="Directory where the project is located"
        )
        session_id: str = SchemaField(
            description="Current session ID"
        )
        summary: str = SchemaField(
            description="Summary of what was accomplished in this session"
        )
        commit_remaining: bool = SchemaField(
            default=True,
            description="Whether to commit any uncommitted changes",
        )

    class Output(Block.Output):
        success: bool = SchemaField(
            description="Whether the session was ended successfully"
        )
        final_commit_hash: str = SchemaField(
            description="Hash of any final commit made",
            default="",
        )
        project_status: dict = SchemaField(
            description="Final project status after this session"
        )
        error: str = SchemaField(
            description="Error message if operation failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="e5f6a7b8-c9d0-1234-ef01-23456789abcd",
            description="End a coding session with a proper summary and commit any remaining changes",
            input_schema=SessionEndBlock.Input,
            output_schema=SessionEndBlock.Output,
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

            # Commit any remaining changes
            commit_hash = ""
            if input_data.commit_remaining:
                commit_hash = manager.git_commit(
                    f"Session {input_data.session_id}: {input_data.summary[:50]}"
                ) or ""

            # End session log
            manager.end_session_log(input_data.session_id, input_data.summary)

            # Update session status
            manager.update_session_status(SessionStatus.PAUSED)

            # Get final status
            project_status = manager.get_project_status()

            yield "success", True
            yield "final_commit_hash", commit_hash
            yield "project_status", project_status

        except Exception as e:
            logger.exception(f"Failed to end session: {e}")
            yield "error", str(e)
            yield "success", False
