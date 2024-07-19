import os
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from google.auth.exceptions import GoogleAuthError
from google.cloud import storage
from google.cloud.exceptions import NotFound

from .gcs import GCSFileStorage, GCSFileStorageConfiguration

try:
    storage.Client()
except GoogleAuthError:
    pytest.skip("Google Cloud Authentication not configured", allow_module_level=True)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def gcs_bucket_name() -> str:
    return f"test-bucket-{str(uuid.uuid4())[:8]}"


@pytest.fixture(scope="module")
def gcs_root() -> Path:
    return Path("/workspaces/AutoGPT-some-unique-task-id")


@pytest.fixture(scope="module")
def gcs_storage_uninitialized(gcs_bucket_name: str, gcs_root: Path):
    os.environ["STORAGE_BUCKET"] = gcs_bucket_name
    storage_config = GCSFileStorageConfiguration.from_env()
    storage_config.root = gcs_root
    storage = GCSFileStorage(storage_config)
    yield storage  # type: ignore
    del os.environ["STORAGE_BUCKET"]


def test_initialize(gcs_bucket_name: str, gcs_storage_uninitialized: GCSFileStorage):
    gcs = gcs_storage_uninitialized._gcs

    # test that the bucket doesn't exist yet
    with pytest.raises(NotFound):
        gcs.get_bucket(gcs_bucket_name)

    gcs_storage_uninitialized.initialize()

    # test that the bucket has been created
    bucket = gcs.get_bucket(gcs_bucket_name)

    # clean up
    bucket.delete(force=True)


@pytest.fixture(scope="module")
def gcs_storage(gcs_storage_uninitialized: GCSFileStorage):
    (gcs_storage := gcs_storage_uninitialized).initialize()
    yield gcs_storage  # type: ignore

    # Empty & delete the test bucket
    gcs_storage._bucket.delete(force=True)


def test_workspace_bucket_name(
    gcs_storage: GCSFileStorage,
    gcs_bucket_name: str,
):
    assert gcs_storage._bucket.name == gcs_bucket_name


NESTED_DIR = "existing/test/dir"
TEST_FILES: list[tuple[str | Path, str]] = [
    ("existing_test_file_1", "test content 1"),
    ("existing_test_file_2.txt", "test content 2"),
    (Path("existing_test_file_3"), "test content 3"),
    (Path(f"{NESTED_DIR}/test_file_4"), "test content 4"),
]


@pytest_asyncio.fixture
async def gcs_storage_with_files(gcs_storage: GCSFileStorage):
    for file_name, file_content in TEST_FILES:
        gcs_storage._bucket.blob(
            str(gcs_storage.get_path(file_name))
        ).upload_from_string(file_content)
    yield gcs_storage  # type: ignore


@pytest.mark.asyncio
async def test_read_file(gcs_storage_with_files: GCSFileStorage):
    for file_name, file_content in TEST_FILES:
        content = gcs_storage_with_files.read_file(file_name)
        assert content == file_content

    with pytest.raises(NotFound):
        gcs_storage_with_files.read_file("non_existent_file")


def test_list_files(gcs_storage_with_files: GCSFileStorage):
    # List at root level
    assert (
        files := gcs_storage_with_files.list_files()
    ) == gcs_storage_with_files.list_files()
    assert len(files) > 0
    assert set(files) == set(Path(file_name) for file_name, _ in TEST_FILES)

    # List at nested path
    assert (
        nested_files := gcs_storage_with_files.list_files(NESTED_DIR)
    ) == gcs_storage_with_files.list_files(NESTED_DIR)
    assert len(nested_files) > 0
    assert set(nested_files) == set(
        p.relative_to(NESTED_DIR)
        for file_name, _ in TEST_FILES
        if (p := Path(file_name)).is_relative_to(NESTED_DIR)
    )


def test_list_folders(gcs_storage_with_files: GCSFileStorage):
    # List recursive
    folders = gcs_storage_with_files.list_folders(recursive=True)
    assert len(folders) > 0
    assert set(folders) == {
        Path("existing"),
        Path("existing/test"),
        Path("existing/test/dir"),
    }
    # List non-recursive
    folders = gcs_storage_with_files.list_folders(recursive=False)
    assert len(folders) > 0
    assert set(folders) == {Path("existing")}


@pytest.mark.asyncio
async def test_write_read_file(gcs_storage: GCSFileStorage):
    await gcs_storage.write_file("test_file", "test_content")
    assert gcs_storage.read_file("test_file") == "test_content"


@pytest.mark.asyncio
async def test_overwrite_file(gcs_storage_with_files: GCSFileStorage):
    for file_name, _ in TEST_FILES:
        await gcs_storage_with_files.write_file(file_name, "new content")
        assert gcs_storage_with_files.read_file(file_name) == "new content"


def test_delete_file(gcs_storage_with_files: GCSFileStorage):
    for file_to_delete, _ in TEST_FILES:
        gcs_storage_with_files.delete_file(file_to_delete)
        assert not gcs_storage_with_files.exists(file_to_delete)


def test_exists(gcs_storage_with_files: GCSFileStorage):
    for file_name, _ in TEST_FILES:
        assert gcs_storage_with_files.exists(file_name)

    assert not gcs_storage_with_files.exists("non_existent_file")


def test_rename_file(gcs_storage_with_files: GCSFileStorage):
    for file_name, _ in TEST_FILES:
        new_name = str(file_name) + "_renamed"
        gcs_storage_with_files.rename(file_name, new_name)
        assert gcs_storage_with_files.exists(new_name)
        assert not gcs_storage_with_files.exists(file_name)


def test_rename_dir(gcs_storage_with_files: GCSFileStorage):
    gcs_storage_with_files.rename(NESTED_DIR, "existing/test/dir_renamed")
    assert gcs_storage_with_files.exists("existing/test/dir_renamed")
    assert not gcs_storage_with_files.exists(NESTED_DIR)


def test_clone(gcs_storage_with_files: GCSFileStorage, gcs_root: Path):
    cloned = gcs_storage_with_files.clone_with_subroot("existing/test")
    assert cloned.root == gcs_root / Path("existing/test")
    assert cloned._bucket.name == gcs_storage_with_files._bucket.name
    assert cloned.exists("dir")
    assert cloned.exists("dir/test_file_4")


@pytest.mark.asyncio
async def test_copy_file(storage: GCSFileStorage):
    await storage.write_file("test_file.txt", "test content")
    storage.copy("test_file.txt", "test_file_copy.txt")
    storage.make_dir("dir")
    storage.copy("test_file.txt", "dir/test_file_copy.txt")
    assert storage.read_file("test_file_copy.txt") == "test content"
    assert storage.read_file("dir/test_file_copy.txt") == "test content"


@pytest.mark.asyncio
async def test_copy_dir(storage: GCSFileStorage):
    storage.make_dir("dir")
    storage.make_dir("dir/sub_dir")
    await storage.write_file("dir/test_file.txt", "test content")
    await storage.write_file("dir/sub_dir/test_file.txt", "test content")
    storage.copy("dir", "dir_copy")
    assert storage.read_file("dir_copy/test_file.txt") == "test content"
    assert storage.read_file("dir_copy/sub_dir/test_file.txt") == "test content"
