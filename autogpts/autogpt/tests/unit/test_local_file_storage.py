from pathlib import Path

import pytest
from forge.file_storage import FileStorageConfiguration, LocalFileStorage

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

_TEST_FILES = [
    Path("test_file.txt"),
    Path("dir/test_file.txt"),
    Path("dir/test_file2.txt"),
    Path("dir/sub_dir/test_file.txt"),
]

_TEST_DIRS = [
    Path("dir"),
    Path("dir/sub_dir"),
]


@pytest.fixture()
def storage_root(tmp_path):
    return tmp_path / "data"


@pytest.fixture()
def storage(storage_root):
    return LocalFileStorage(
        FileStorageConfiguration(root=storage_root, restrict_to_root=True)
    )


@pytest.fixture()
def content():
    return "test content"


@pytest.fixture(params=_ACCESSIBLE_PATHS)
def accessible_path(request):
    return request.param


@pytest.fixture(params=_INACCESSIBLE_PATHS)
def inaccessible_path(request):
    return request.param


@pytest.fixture(params=_TEST_FILES)
def file_path(request):
    return request.param


@pytest.mark.asyncio
async def test_open_file(file_path: Path, content: str, storage: LocalFileStorage):
    if file_path.parent:
        storage.make_dir(file_path.parent)
    await storage.write_file(file_path, content)
    file = storage.open_file(file_path)
    assert file.read() == content
    file.close()
    storage.delete_file(file_path)


@pytest.mark.asyncio
async def test_write_read_file(content: str, storage: LocalFileStorage):
    await storage.write_file("test_file.txt", content)
    assert storage.read_file("test_file.txt") == content


@pytest.mark.asyncio
async def test_list_files(content: str, storage: LocalFileStorage):
    storage.make_dir("dir")
    storage.make_dir("dir/sub_dir")
    await storage.write_file("test_file.txt", content)
    await storage.write_file("dir/test_file.txt", content)
    await storage.write_file("dir/test_file2.txt", content)
    await storage.write_file("dir/sub_dir/test_file.txt", content)
    files = storage.list_files()
    assert Path("test_file.txt") in files
    assert Path("dir/test_file.txt") in files
    assert Path("dir/test_file2.txt") in files
    assert Path("dir/sub_dir/test_file.txt") in files
    storage.delete_file("test_file.txt")
    storage.delete_file("dir/test_file.txt")
    storage.delete_file("dir/test_file2.txt")
    storage.delete_file("dir/sub_dir/test_file.txt")
    storage.delete_dir("dir/sub_dir")
    storage.delete_dir("dir")


@pytest.mark.asyncio
async def test_list_folders(content: str, storage: LocalFileStorage):
    storage.make_dir("dir")
    storage.make_dir("dir/sub_dir")
    await storage.write_file("dir/test_file.txt", content)
    await storage.write_file("dir/sub_dir/test_file.txt", content)
    folders = storage.list_folders(recursive=False)
    folders_recursive = storage.list_folders(recursive=True)
    assert Path("dir") in folders
    assert Path("dir/sub_dir") not in folders
    assert Path("dir") in folders_recursive
    assert Path("dir/sub_dir") in folders_recursive
    storage.delete_file("dir/test_file.txt")
    storage.delete_file("dir/sub_dir/test_file.txt")
    storage.delete_dir("dir/sub_dir")
    storage.delete_dir("dir")


@pytest.mark.asyncio
async def test_exists_delete_file(
    file_path: Path, content: str, storage: LocalFileStorage
):
    if file_path.parent:
        storage.make_dir(file_path.parent)
    await storage.write_file(file_path, content)
    assert storage.exists(file_path)
    storage.delete_file(file_path)
    assert not storage.exists(file_path)


@pytest.fixture(params=_TEST_DIRS)
def test_make_delete_dir(request, storage: LocalFileStorage):
    storage.make_dir(request)
    assert storage.exists(request)
    storage.delete_dir(request)
    assert not storage.exists(request)


@pytest.mark.asyncio
async def test_rename(file_path: Path, content: str, storage: LocalFileStorage):
    if file_path.parent:
        storage.make_dir(file_path.parent)
    await storage.write_file(file_path, content)
    assert storage.exists(file_path)
    storage.rename(file_path, Path(str(file_path) + "_renamed"))
    assert not storage.exists(file_path)
    assert storage.exists(Path(str(file_path) + "_renamed"))


def test_clone_with_subroot(storage: LocalFileStorage):
    subroot = storage.clone_with_subroot("dir")
    assert subroot.root == storage.root / "dir"


def test_get_path_accessible(accessible_path: Path, storage: LocalFileStorage):
    full_path = storage.get_path(accessible_path)
    assert full_path.is_absolute()
    assert full_path.is_relative_to(storage.root)


def test_get_path_inaccessible(inaccessible_path: Path, storage: LocalFileStorage):
    with pytest.raises(ValueError):
        storage.get_path(inaccessible_path)


@pytest.mark.asyncio
async def test_copy_file(storage: LocalFileStorage):
    await storage.write_file("test_file.txt", "test content")
    storage.copy("test_file.txt", "test_file_copy.txt")
    storage.make_dir("dir")
    storage.copy("test_file.txt", "dir/test_file_copy.txt")
    assert storage.read_file("test_file_copy.txt") == "test content"
    assert storage.read_file("dir/test_file_copy.txt") == "test content"


@pytest.mark.asyncio
async def test_copy_dir(storage: LocalFileStorage):
    storage.make_dir("dir")
    storage.make_dir("dir/sub_dir")
    await storage.write_file("dir/test_file.txt", "test content")
    await storage.write_file("dir/sub_dir/test_file.txt", "test content")
    storage.copy("dir", "dir_copy")
    assert storage.read_file("dir_copy/test_file.txt") == "test content"
    assert storage.read_file("dir_copy/sub_dir/test_file.txt") == "test content"
