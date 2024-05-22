import os
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from botocore.exceptions import ClientError
from forge.file_storage import S3FileStorage, S3FileStorageConfiguration

if not (os.getenv("S3_ENDPOINT_URL") and os.getenv("AWS_ACCESS_KEY_ID")):
    pytest.skip("S3 environment variables are not set", allow_module_level=True)


@pytest.fixture
def s3_bucket_name() -> str:
    return f"test-bucket-{str(uuid.uuid4())[:8]}"


@pytest.fixture
def s3_root() -> Path:
    return Path("/workspaces/AutoGPT-some-unique-task-id")


@pytest.fixture
def s3_storage_uninitialized(s3_bucket_name: str, s3_root: Path) -> S3FileStorage:
    os.environ["STORAGE_BUCKET"] = s3_bucket_name
    storage_config = S3FileStorageConfiguration.from_env()
    storage_config.root = s3_root
    storage = S3FileStorage(storage_config)
    yield storage  # type: ignore
    del os.environ["STORAGE_BUCKET"]


def test_initialize(s3_bucket_name: str, s3_storage_uninitialized: S3FileStorage):
    s3 = s3_storage_uninitialized._s3

    # test that the bucket doesn't exist yet
    with pytest.raises(ClientError):
        s3.meta.client.head_bucket(Bucket=s3_bucket_name)

    s3_storage_uninitialized.initialize()

    # test that the bucket has been created
    s3.meta.client.head_bucket(Bucket=s3_bucket_name)


def test_workspace_bucket_name(
    s3_storage: S3FileStorage,
    s3_bucket_name: str,
):
    assert s3_storage._bucket.name == s3_bucket_name


@pytest.fixture
def s3_storage(s3_storage_uninitialized: S3FileStorage) -> S3FileStorage:
    (s3_storage := s3_storage_uninitialized).initialize()
    yield s3_storage  # type: ignore

    # Empty & delete the test bucket
    s3_storage._bucket.objects.all().delete()
    s3_storage._bucket.delete()


NESTED_DIR = "existing/test/dir"
TEST_FILES: list[tuple[str | Path, str]] = [
    ("existing_test_file_1", "test content 1"),
    ("existing_test_file_2.txt", "test content 2"),
    (Path("existing_test_file_3"), "test content 3"),
    (Path(f"{NESTED_DIR}/test_file_4"), "test content 4"),
]


@pytest_asyncio.fixture
async def s3_storage_with_files(s3_storage: S3FileStorage) -> S3FileStorage:
    for file_name, file_content in TEST_FILES:
        s3_storage._bucket.Object(str(s3_storage.get_path(file_name))).put(
            Body=file_content
        )
    yield s3_storage  # type: ignore


@pytest.mark.asyncio
async def test_read_file(s3_storage_with_files: S3FileStorage):
    for file_name, file_content in TEST_FILES:
        content = s3_storage_with_files.read_file(file_name)
        assert content == file_content

    with pytest.raises(ClientError):
        s3_storage_with_files.read_file("non_existent_file")


def test_list_files(s3_storage_with_files: S3FileStorage):
    # List at root level
    assert (
        files := s3_storage_with_files.list_files()
    ) == s3_storage_with_files.list_files()
    assert len(files) > 0
    assert set(files) == set(Path(file_name) for file_name, _ in TEST_FILES)

    # List at nested path
    assert (
        nested_files := s3_storage_with_files.list_files(NESTED_DIR)
    ) == s3_storage_with_files.list_files(NESTED_DIR)
    assert len(nested_files) > 0
    assert set(nested_files) == set(
        p.relative_to(NESTED_DIR)
        for file_name, _ in TEST_FILES
        if (p := Path(file_name)).is_relative_to(NESTED_DIR)
    )


def test_list_folders(s3_storage_with_files: S3FileStorage):
    # List recursive
    folders = s3_storage_with_files.list_folders(recursive=True)
    assert len(folders) > 0
    assert set(folders) == {
        Path("existing"),
        Path("existing/test"),
        Path("existing/test/dir"),
    }
    # List non-recursive
    folders = s3_storage_with_files.list_folders(recursive=False)
    assert len(folders) > 0
    assert set(folders) == {Path("existing")}


@pytest.mark.asyncio
async def test_write_read_file(s3_storage: S3FileStorage):
    await s3_storage.write_file("test_file", "test_content")
    assert s3_storage.read_file("test_file") == "test_content"


@pytest.mark.asyncio
async def test_overwrite_file(s3_storage_with_files: S3FileStorage):
    for file_name, _ in TEST_FILES:
        await s3_storage_with_files.write_file(file_name, "new content")
        assert s3_storage_with_files.read_file(file_name) == "new content"


def test_delete_file(s3_storage_with_files: S3FileStorage):
    for file_to_delete, _ in TEST_FILES:
        s3_storage_with_files.delete_file(file_to_delete)
        with pytest.raises(ClientError):
            s3_storage_with_files.read_file(file_to_delete)


def test_exists(s3_storage_with_files: S3FileStorage):
    for file_name, _ in TEST_FILES:
        assert s3_storage_with_files.exists(file_name)

    assert not s3_storage_with_files.exists("non_existent_file")


def test_rename_file(s3_storage_with_files: S3FileStorage):
    for file_name, _ in TEST_FILES:
        new_name = str(file_name) + "_renamed"
        s3_storage_with_files.rename(file_name, new_name)
        assert s3_storage_with_files.exists(new_name)
        assert not s3_storage_with_files.exists(file_name)


def test_rename_dir(s3_storage_with_files: S3FileStorage):
    s3_storage_with_files.rename(NESTED_DIR, "existing/test/dir_renamed")
    assert s3_storage_with_files.exists("existing/test/dir_renamed")
    assert not s3_storage_with_files.exists(NESTED_DIR)


def test_clone(s3_storage_with_files: S3FileStorage, s3_root: Path):
    cloned = s3_storage_with_files.clone_with_subroot("existing/test")
    assert cloned.root == s3_root / Path("existing/test")
    assert cloned._bucket.name == s3_storage_with_files._bucket.name
    assert cloned.exists("dir")
    assert cloned.exists("dir/test_file_4")


@pytest.mark.asyncio
async def test_copy_file(storage: S3FileStorage):
    await storage.write_file("test_file.txt", "test content")
    storage.copy("test_file.txt", "test_file_copy.txt")
    storage.make_dir("dir")
    storage.copy("test_file.txt", "dir/test_file_copy.txt")
    assert storage.read_file("test_file_copy.txt") == "test content"
    assert storage.read_file("dir/test_file_copy.txt") == "test content"


@pytest.mark.asyncio
async def test_copy_dir(storage: S3FileStorage):
    storage.make_dir("dir")
    storage.make_dir("dir/sub_dir")
    await storage.write_file("dir/test_file.txt", "test content")
    await storage.write_file("dir/sub_dir/test_file.txt", "test content")
    storage.copy("dir", "dir_copy")
    assert storage.read_file("dir_copy/test_file.txt") == "test content"
    assert storage.read_file("dir_copy/sub_dir/test_file.txt") == "test content"
