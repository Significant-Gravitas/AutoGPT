import pytest
from git.exc import GitCommandError
from git.repo.base import Repo

from forge.file_storage.base import FileStorage
from forge.utils.exceptions import CommandExecutionError

from . import GitOperationsComponent


@pytest.fixture
def mock_clone_from(mocker):
    return mocker.patch.object(Repo, "clone_from")


@pytest.fixture
def git_ops_component():
    return GitOperationsComponent()


def test_clone_auto_gpt_repository(
    git_ops_component: GitOperationsComponent,
    storage: FileStorage,
    mock_clone_from,
):
    mock_clone_from.return_value = None

    repo = "github.com/Significant-Gravitas/Auto-GPT.git"
    scheme = "https://"
    url = scheme + repo
    clone_path = storage.get_path("auto-gpt-repo")

    expected_output = f"Cloned {url} to {clone_path}"

    clone_result = git_ops_component.clone_repository(url, clone_path)

    assert clone_result == expected_output
    mock_clone_from.assert_called_once_with(
        url=f"{scheme}{git_ops_component.config.github_username}:{git_ops_component.config.github_api_key}@{repo}",  # noqa: E501
        to_path=clone_path,
    )


def test_clone_repository_error(
    git_ops_component: GitOperationsComponent,
    storage: FileStorage,
    mock_clone_from,
):
    url = "https://github.com/this-repository/does-not-exist.git"
    clone_path = storage.get_path("does-not-exist")

    mock_clone_from.side_effect = GitCommandError(
        "clone", "fatal: repository not found", ""
    )

    with pytest.raises(CommandExecutionError):
        git_ops_component.clone_repository(url, clone_path)


def test_clone_github_host_injects_credentials(
    git_ops_component: GitOperationsComponent,
    storage: FileStorage,
    mock_clone_from,
):
    mock_clone_from.return_value = None

    url = "https://github.com/owner/repo.git"
    clone_path = storage.get_path("github-repo")

    git_ops_component.clone_repository(url, clone_path)

    mock_clone_from.assert_called_once()
    called_url = mock_clone_from.call_args.kwargs["url"]
    assert "github.com/owner/repo.git" in called_url
    assert called_url.startswith(
        f"https://{git_ops_component.config.github_username}:"
        f"{git_ops_component.config.github_api_key}@github.com"
    )


def test_clone_non_github_host_is_refused(
    git_ops_component: GitOperationsComponent,
    storage: FileStorage,
    mock_clone_from,
):
    url = "https://evil.example.com/repo.git"
    clone_path = storage.get_path("evil-repo")

    with pytest.raises(CommandExecutionError):
        git_ops_component.clone_repository(url, clone_path)

    mock_clone_from.assert_not_called()
