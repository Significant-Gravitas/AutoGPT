import os
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from google.cloud.exceptions import NotFound

from autogpt.file_workspace.gcs import GCSFileWorkspace, GCSFileWorkspaceConfiguration

if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    pytest.skip("GOOGLE_APPLICATION_CREDENTIALS are not set", allow_module_level=True)


@pytest.fixture
def gcs_bucket_name() -> str:
    return f"test-bucket-{str(uuid.uuid4())[:8]}"


@pytest.fixture
def gcs_workspace_uninitialized(gcs_bucket_name: str) -> GCSFileWorkspace:
    os.environ["WORKSPACE_STORAGE_BUCKET"] = gcs_bucket_name
    ws_config = GCSFileWorkspaceConfiguration.from_env()
    workspace = GCSFileWorkspace(ws_config)
    yield workspace  # type: ignore
    del os.environ["WORKSPACE_STORAGE_BUCKET"]


def test_initialize(
    gcs_bucket_name: str, gcs_workspace_uninitialized: GCSFileWorkspace
):
    gcs = gcs_workspace_uninitialized._bucket

    # test that the bucket doesn't exist yet
    with pytest.raises(NotFound):
        gcs.get_blob(gcs_bucket_name)

    gcs_workspace_uninitialized.initialize()

    # test that the bucket has been created
    gcs.get_blob(gcs_bucket_name)


def test_workspace_bucket_name(
    gcs_workspace: GCSFileWorkspace,
    gcs_bucket_name: str,
):
    assert gcs_workspace._bucket.name == gcs_bucket_name


@pytest.fixture
def gcs_workspace(gcs_workspace_uninitialized: GCSFileWorkspace) -> GCSFileWorkspace:
    (gcs_workspace := gcs_workspace_uninitialized).initialize()
    yield gcs_workspace  # type: ignore

    # Empty & delete the test bucket
    gcs_workspace._bucket.delete_blobs(gcs_workspace._bucket.list_blobs())
    gcs_workspace._bucket.delete()


TEST_FILES: list[tuple[str | Path, str]] = [
    ("existing_test_file_1", "test content 1"),
    ("/existing_test_file_2.txt", "test content 2"),
    (Path("/existing_test_file_3"), "test content 3"),
    (Path("existing/test/file/4"), "test content 4"),
]


@pytest_asyncio.fixture
async def gcs_workspace_with_files(gcs_workspace: GCSFileWorkspace) -> GCSFileWorkspace:
    for file_name, file_content in TEST_FILES:
        gcs_workspace._bucket.blob(
            str(gcs_workspace.get_path(file_name))
        ).upload_from_string(file_content)
    yield gcs_workspace  # type: ignore


@pytest.mark.asyncio
async def test_read_file(gcs_workspace_with_files: GCSFileWorkspace):
    for file_name, file_content in TEST_FILES:
        content = gcs_workspace_with_files.read_file(file_name)
        assert content == file_content

    with pytest.raises(NotFound):
        gcs_workspace_with_files.read_file("non_existent_file")


def test_list_files(gcs_workspace_with_files: GCSFileWorkspace):
    # List at root level
    assert (files := gcs_workspace_with_files.list()) == gcs_workspace_with_files.list()
    assert len(files) > 0
    assert set(files) == set(
        p.relative_to("/") if (p := Path(file_name)).is_absolute() else p
        for file_name, _ in TEST_FILES
    )

    # List at nested path
    assert (
        nested_files := gcs_workspace_with_files.list("existing")
    ) == gcs_workspace_with_files.list("existing")
    assert len(nested_files) > 0
    assert set(nested_files) == set(
        p
        for file_name, _ in TEST_FILES
        if (p := Path(file_name)).is_relative_to("existing")
    )


@pytest.mark.asyncio
async def test_write_read_file(gcs_workspace: GCSFileWorkspace):
    await gcs_workspace.write_file("test_file", "test_content")
    assert gcs_workspace.read_file("test_file") == "test_content"


@pytest.mark.asyncio
async def test_overwrite_file(gcs_workspace_with_files: GCSFileWorkspace):
    for file_name, _ in TEST_FILES:
        await gcs_workspace_with_files.write_file(file_name, "new content")
        assert gcs_workspace_with_files.read_file(file_name) == "new content"


def test_delete_file(gcs_workspace_with_files: GCSFileWorkspace):
    for file_to_delete, _ in TEST_FILES:
        gcs_workspace_with_files.delete_file(file_to_delete)
        with pytest.raises(NotFound):
            gcs_workspace_with_files.read_file(file_to_delete)
