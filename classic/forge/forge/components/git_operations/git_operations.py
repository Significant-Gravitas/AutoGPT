from pathlib import Path
from typing import Iterator, Optional

from git.exc import GitCommandError, InvalidGitRepositoryError
from git.repo import Repo
from pydantic import BaseModel, SecretStr

from forge.agent.components import ConfigurableComponent
from forge.agent.protocols import CommandProvider
from forge.command import Command, command
from forge.models.config import UserConfigurable
from forge.models.json_schema import JSONSchema
from forge.utils.exceptions import CommandExecutionError
from forge.utils.url_validator import validate_url


class GitOperationsConfiguration(BaseModel):
    github_username: Optional[str] = UserConfigurable(None, from_env="GITHUB_USERNAME")
    github_api_key: Optional[SecretStr] = UserConfigurable(
        None, from_env="GITHUB_API_KEY", exclude=True
    )


class GitOperationsComponent(
    CommandProvider, ConfigurableComponent[GitOperationsConfiguration]
):
    """Provides commands to perform Git operations."""

    config_class = GitOperationsConfiguration

    def __init__(self, config: Optional[GitOperationsConfiguration] = None):
        ConfigurableComponent.__init__(self, config)
        # Clone repository needs credentials, but other git operations work without
        self._enabled = True

    def get_commands(self) -> Iterator[Command]:
        # Only yield clone if credentials are configured
        if self.config.github_username and self.config.github_api_key:
            yield self.clone_repository
        # These commands work on any local git repository
        yield self.git_status
        yield self.git_add
        yield self.git_commit
        yield self.git_push
        yield self.git_pull
        yield self.git_diff
        yield self.git_branch
        yield self.git_checkout
        yield self.git_log

    def _get_repo(self, repo_path: str | Path | None = None) -> Repo:
        """Get a Repo object for the given path.

        Args:
            repo_path: Path to the repository, or None for current directory

        Returns:
            Repo: The git repository object

        Raises:
            CommandExecutionError: If the path is not a git repository
        """
        path = Path(repo_path) if repo_path else Path.cwd()
        try:
            return Repo(path, search_parent_directories=True)
        except InvalidGitRepositoryError:
            raise CommandExecutionError(
                f"'{path}' is not a git repository (or any parent up to mount point)"
            )

    @command(
        parameters={
            "url": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The URL of the repository to clone",
                required=True,
            ),
            "clone_path": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The path to clone the repository to",
                required=True,
            ),
        },
    )
    @validate_url
    def clone_repository(self, url: str, clone_path: Path) -> str:
        """Clone a GitHub repository locally.

        Args:
            url (str): The URL of the repository to clone.
            clone_path (Path): The path to clone the repository to.

        Returns:
            str: The result of the clone operation.
        """
        split_url = url.split("//")
        api_key = (
            self.config.github_api_key.get_secret_value()
            if self.config.github_api_key
            else None
        )
        auth_repo_url = f"//{self.config.github_username}:" f"{api_key}@".join(
            split_url
        )
        try:
            Repo.clone_from(url=auth_repo_url, to_path=clone_path)
        except Exception as e:
            raise CommandExecutionError(f"Could not clone repo: {e}")

        return f"""Cloned {url} to {clone_path}"""

    @command(
        ["git_status"],
        "Show the working tree status including staged, unstaged, and untracked files.",
        {
            "repo_path": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Path to the repository (default: current directory)",
                required=False,
            ),
        },
    )
    def git_status(self, repo_path: str | None = None) -> str:
        """Show the working tree status.

        Args:
            repo_path: Path to the repository

        Returns:
            str: Status information
        """
        repo = self._get_repo(repo_path)

        # Get the current branch
        try:
            branch = repo.active_branch.name
        except TypeError:
            branch = "HEAD detached"

        # Get status information
        staged = [item.a_path for item in repo.index.diff("HEAD")]
        unstaged = [item.a_path for item in repo.index.diff(None)]
        untracked = repo.untracked_files

        lines = [f"On branch {branch}", ""]

        if staged:
            lines.append("Changes to be committed:")
            for file in staged:
                lines.append(f"  modified: {file}")
            lines.append("")

        if unstaged:
            lines.append("Changes not staged for commit:")
            for file in unstaged:
                lines.append(f"  modified: {file}")
            lines.append("")

        if untracked:
            lines.append("Untracked files:")
            for file in untracked:
                lines.append(f"  {file}")
            lines.append("")

        if not staged and not unstaged and not untracked:
            lines.append("nothing to commit, working tree clean")

        return "\n".join(lines)

    @command(
        ["git_add", "stage_files"],
        "Stage files for commit.",
        {
            "files": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                items=JSONSchema(type=JSONSchema.Type.STRING),
                description="Files to stage. Use ['.'] to stage all changes.",
                required=True,
            ),
            "repo_path": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Path to the repository (default: current directory)",
                required=False,
            ),
        },
    )
    def git_add(self, files: list[str], repo_path: str | None = None) -> str:
        """Stage files for commit.

        Args:
            files: List of files to stage
            repo_path: Path to the repository

        Returns:
            str: Confirmation message
        """
        repo = self._get_repo(repo_path)

        try:
            if files == ["."]:
                repo.git.add(A=True)
                return "Staged all changes"
            else:
                repo.index.add(files)
                return f"Staged files: {', '.join(files)}"
        except GitCommandError as e:
            raise CommandExecutionError(f"Failed to stage files: {e}")

    @command(
        ["git_commit"],
        "Commit staged changes with a message.",
        {
            "message": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The commit message",
                required=True,
            ),
            "repo_path": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Path to the repository (default: current directory)",
                required=False,
            ),
        },
    )
    def git_commit(self, message: str, repo_path: str | None = None) -> str:
        """Commit staged changes.

        Args:
            message: The commit message
            repo_path: Path to the repository

        Returns:
            str: Confirmation with commit hash
        """
        repo = self._get_repo(repo_path)

        # Check if there are staged changes
        if not repo.index.diff("HEAD"):
            raise CommandExecutionError("Nothing to commit (no staged changes)")

        try:
            commit = repo.index.commit(message)
            return f"Committed: {commit.hexsha[:8]} - {message}"
        except GitCommandError as e:
            raise CommandExecutionError(f"Failed to commit: {e}")

    @command(
        ["git_push"],
        "Push commits to a remote repository.",
        {
            "repo_path": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Path to the repository (default: current directory)",
                required=False,
            ),
            "remote": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Remote name (default: origin)",
                required=False,
            ),
            "branch": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Branch to push (default: current branch)",
                required=False,
            ),
        },
    )
    def git_push(
        self,
        repo_path: str | None = None,
        remote: str = "origin",
        branch: str | None = None,
    ) -> str:
        """Push commits to remote.

        Args:
            repo_path: Path to the repository
            remote: Remote name
            branch: Branch to push

        Returns:
            str: Confirmation message
        """
        repo = self._get_repo(repo_path)

        try:
            if branch is None:
                branch = repo.active_branch.name
        except TypeError:
            raise CommandExecutionError("Cannot push from detached HEAD state")

        try:
            push_info = repo.remote(remote).push(branch)
            if push_info:
                return f"Pushed {branch} to {remote}"
            return f"Pushed {branch} to {remote}"
        except GitCommandError as e:
            raise CommandExecutionError(f"Failed to push: {e}")

    @command(
        ["git_pull"],
        "Pull changes from a remote repository.",
        {
            "repo_path": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Path to the repository (default: current directory)",
                required=False,
            ),
            "remote": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Remote name (default: origin)",
                required=False,
            ),
            "branch": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Branch to pull (default: current branch)",
                required=False,
            ),
        },
    )
    def git_pull(
        self,
        repo_path: str | None = None,
        remote: str = "origin",
        branch: str | None = None,
    ) -> str:
        """Pull changes from remote.

        Args:
            repo_path: Path to the repository
            remote: Remote name
            branch: Branch to pull

        Returns:
            str: Result of the pull operation
        """
        repo = self._get_repo(repo_path)

        try:
            if branch is None:
                branch = repo.active_branch.name
        except TypeError:
            raise CommandExecutionError("Cannot pull in detached HEAD state")

        try:
            pull_info = repo.remote(remote).pull(branch)
            if pull_info:
                return f"Pulled {branch} from {remote}: {pull_info[0].note}"
            return f"Pulled {branch} from {remote}"
        except GitCommandError as e:
            raise CommandExecutionError(f"Failed to pull: {e}")

    @command(
        ["git_diff"],
        "Show changes between commits, working tree, etc.",
        {
            "repo_path": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Path to the repository (default: current directory)",
                required=False,
            ),
            "staged": JSONSchema(
                type=JSONSchema.Type.BOOLEAN,
                description="Show staged changes only (default: False)",
                required=False,
            ),
            "file": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Specific file to diff (default: all files)",
                required=False,
            ),
        },
    )
    def git_diff(
        self,
        repo_path: str | None = None,
        staged: bool = False,
        file: str | None = None,
    ) -> str:
        """Show changes in the repository.

        Args:
            repo_path: Path to the repository
            staged: Show only staged changes
            file: Specific file to diff

        Returns:
            str: The diff output
        """
        repo = self._get_repo(repo_path)

        try:
            if staged:
                diff = (
                    repo.git.diff("--cached", file)
                    if file
                    else repo.git.diff("--cached")
                )
            else:
                diff = repo.git.diff(file) if file else repo.git.diff()

            if not diff:
                return "No changes" + (" in staged files" if staged else "")

            return diff
        except GitCommandError as e:
            raise CommandExecutionError(f"Failed to get diff: {e}")

    @command(
        ["git_branch"],
        "List, create, or delete branches.",
        {
            "repo_path": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Path to the repository (default: current directory)",
                required=False,
            ),
            "name": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Name of the branch to create (omit to list branches)",
                required=False,
            ),
            "delete": JSONSchema(
                type=JSONSchema.Type.BOOLEAN,
                description="Delete the specified branch (default: False)",
                required=False,
            ),
        },
    )
    def git_branch(
        self,
        repo_path: str | None = None,
        name: str | None = None,
        delete: bool = False,
    ) -> str:
        """List, create, or delete branches.

        Args:
            repo_path: Path to the repository
            name: Branch name to create/delete
            delete: Whether to delete the branch

        Returns:
            str: Result of the operation
        """
        repo = self._get_repo(repo_path)

        try:
            if name is None:
                # List branches
                branches = []
                current = repo.active_branch.name if not repo.head.is_detached else None
                for branch in repo.branches:
                    prefix = "* " if branch.name == current else "  "
                    branches.append(f"{prefix}{branch.name}")
                return "\n".join(branches) if branches else "No branches found"

            if delete:
                # Delete branch
                repo.delete_head(name, force=True)
                return f"Deleted branch '{name}'"
            else:
                # Create branch
                repo.create_head(name)
                return f"Created branch '{name}'"

        except GitCommandError as e:
            raise CommandExecutionError(f"Branch operation failed: {e}")

    @command(
        ["git_checkout"],
        "Switch branches or restore working tree files.",
        {
            "target": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Branch name or commit to checkout",
                required=True,
            ),
            "repo_path": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Path to the repository (default: current directory)",
                required=False,
            ),
            "create_branch": JSONSchema(
                type=JSONSchema.Type.BOOLEAN,
                description="Create a new branch with the given name (default: False)",
                required=False,
            ),
        },
    )
    def git_checkout(
        self,
        target: str,
        repo_path: str | None = None,
        create_branch: bool = False,
    ) -> str:
        """Checkout a branch or commit.

        Args:
            target: Branch or commit to checkout
            repo_path: Path to the repository
            create_branch: Whether to create a new branch

        Returns:
            str: Confirmation message
        """
        repo = self._get_repo(repo_path)

        try:
            if create_branch:
                # Create and checkout new branch
                new_branch = repo.create_head(target)
                new_branch.checkout()
                return f"Switched to new branch '{target}'"
            else:
                # Checkout existing branch or commit
                repo.git.checkout(target)
                return f"Switched to '{target}'"

        except GitCommandError as e:
            raise CommandExecutionError(f"Checkout failed: {e}")

    @command(
        ["git_log"],
        "Show commit logs.",
        {
            "repo_path": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Path to the repository (default: current directory)",
                required=False,
            ),
            "max_count": JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="Maximum number of commits to show (default: 10)",
                required=False,
            ),
            "oneline": JSONSchema(
                type=JSONSchema.Type.BOOLEAN,
                description="Use single-line format (default: False)",
                required=False,
            ),
        },
    )
    def git_log(
        self,
        repo_path: str | None = None,
        max_count: int = 10,
        oneline: bool = False,
    ) -> str:
        """Show commit history.

        Args:
            repo_path: Path to the repository
            max_count: Maximum commits to show
            oneline: Use single-line format

        Returns:
            str: Commit log
        """
        repo = self._get_repo(repo_path)

        try:
            commits = list(repo.iter_commits(max_count=max_count))
            if not commits:
                return "No commits found"

            lines = []
            for commit in commits:
                if oneline:
                    lines.append(f"{commit.hexsha[:8]} {commit.summary}")
                else:
                    lines.append(f"commit {commit.hexsha}")
                    lines.append(
                        f"Author: {commit.author.name} <{commit.author.email}>"
                    )
                    lines.append(f"Date:   {commit.committed_datetime}")
                    lines.append("")
                    lines.append(f"    {commit.message.strip()}")
                    lines.append("")

            return "\n".join(lines)

        except GitCommandError as e:
            raise CommandExecutionError(f"Failed to get log: {e}")
