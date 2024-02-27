from pathlib import Path

import pytest

from autogpt.file_storage.local import FileStorageConfiguration, LocalFileStorage

_ACCESSIBLE_PATHS = [
    Path("."),
    Path("test_file.txt"),
    Path("test_folder"),
    Path("test_folder/test_file.txt"),
    Path("test_folder/.."),
    Path("test_folder/../test_file.txt"),
    Path("test_folder/../test_folder"),
    Path("test_folder/../test_folder/test_file.txt"),
]

_INACCESSIBLE_PATHS = (
    [
        # Takes us out of the workspace
        Path(".."),
        Path("../test_file.txt"),
        Path("../not_auto_gpt_workspace"),
        Path("../not_auto_gpt_workspace/test_file.txt"),
        Path("test_folder/../.."),
        Path("test_folder/../../test_file.txt"),
        Path("test_folder/../../not_auto_gpt_workspace"),
        Path("test_folder/../../not_auto_gpt_workspace/test_file.txt"),
    ]
    + [
        # Contains null byte
        Path("\0"),
        Path("\0test_file.txt"),
        Path("test_folder/\0"),
        Path("test_folder/\0test_file.txt"),
    ]
    + [
        # Absolute paths
        Path("/"),
        Path("/test_file.txt"),
        Path("/home"),
    ]
)


@pytest.fixture()
def storage_root(tmp_path):
    return tmp_path / "data"


@pytest.fixture()
def storage(storage_root):
    return LocalFileStorage(
        FileStorageConfiguration(root=storage_root, restrict_to_root=True)
    )


@pytest.fixture(params=_ACCESSIBLE_PATHS)
def accessible_path(request):
    return request.param


@pytest.fixture(params=_INACCESSIBLE_PATHS)
def inaccessible_path(request):
    return request.param


def test_get_path_accessible(accessible_path: Path, storage: LocalFileStorage):
    full_path = storage.get_path(accessible_path)
    assert full_path.is_absolute()
    assert full_path.is_relative_to(storage.root)


def test_get_path_inaccessible(inaccessible_path: Path, storage: LocalFileStorage):
    with pytest.raises(ValueError):
        storage.get_path(inaccessible_path)
