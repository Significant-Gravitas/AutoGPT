"""Files of one name staged in one run keep their own bytes (#17280).

``store_media_file`` names the local copy it makes after the file. A workflow
run keeps every node's copies in one folder, so two files of one name each
need a path of their own there, or the second copy replaces the first. The
path a staging returns still works as a local path handed back to it."""

import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from backend.blocks.basic import FileStoreBlock
from backend.data.execution import ExecutionContext
from backend.util import file as file_module
from backend.util.file import get_exec_file_path, separate_exec_files, store_media_file
from backend.util.type import MediaFileType

_EXEC_ID = "exec-1"
_CONTEXT = ExecutionContext(
    user_id="user-1", graph_exec_id=_EXEC_ID, workspace_id="workspace-1"
)

# Two files called report.csv from each kind of source a copy is named after.
_SAME_NAME = {
    "workspace": ("workspace://file-a#text/csv", "workspace://file-b#text/csv"),
    "url": (
        "https://a.example/reports/report.csv",
        "https://b.example/archive/report.csv",
    ),
    "cloud": ("gcs://bucket-a/reports/report.csv", "gcs://bucket-b/archive/report.csv"),
}
_CONTENT = {
    reference: f"bytes of {reference}".encode()
    for references in _SAME_NAME.values()
    for reference in references
}
_REPORT = _SAME_NAME["workspace"][0]


@pytest.fixture(autouse=True)
def exec_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The exec_file folder under ``tmp_path``, and each URL and cloud file in
    ``_SAME_NAME`` serving its own bytes."""
    monkeypatch.setattr(file_module, "TEMP_DIR", tmp_path.resolve())
    monkeypatch.setattr(file_module, "scan_content_safe", AsyncMock())

    downloads = MagicMock()
    downloads.return_value.get = AsyncMock(
        side_effect=lambda url: SimpleNamespace(content=_CONTENT[url], headers={})
    )
    monkeypatch.setattr(file_module, "Requests", downloads)

    storage = MagicMock()
    storage.is_cloud_path.side_effect = lambda path: path.startswith("gcs://")
    storage.parse_cloud_path.side_effect = lambda path: (
        "gcs",
        path.removeprefix("gcs://"),
    )
    storage.retrieve_file = AsyncMock(side_effect=lambda path, **_: _CONTENT[path])
    monkeypatch.setattr(
        file_module, "get_cloud_storage_handler", AsyncMock(return_value=storage)
    )
    return tmp_path.resolve() / "exec_file"


@pytest.fixture(autouse=True)
def workspace(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """The run's workspace: each of its files in ``_SAME_NAME`` holds its own
    bytes, and a file stored in it comes back as ``workspace://stored``."""
    workspace = MagicMock()
    workspace.read_file_by_id = AsyncMock(
        side_effect=lambda file_id: _CONTENT[f"workspace://{file_id}#text/csv"]
    )
    workspace.get_file_info = AsyncMock(return_value=SimpleNamespace(name="report.csv"))
    workspace.write_file = AsyncMock(
        return_value=SimpleNamespace(id="stored", mime_type="text/csv")
    )
    monkeypatch.setattr(
        "backend.util.workspace.WorkspaceManager", MagicMock(return_value=workspace)
    )
    return workspace


@pytest.mark.parametrize("source", list(_SAME_NAME))
async def test_two_files_of_one_name_staged_in_one_run_keep_their_own_bytes(
    source: str, exec_files: Path
) -> None:
    first, second = _SAME_NAME[source]

    first_copy = await _stage(first)
    second_copy = await _stage(second)

    assert first_copy != second_copy
    assert first_copy.name == second_copy.name == "report.csv"
    assert first_copy.read_bytes() == _CONTENT[first]
    assert second_copy.read_bytes() == _CONTENT[second]
    assert first_copy.is_relative_to(exec_files / _EXEC_ID)
    assert second_copy.is_relative_to(exec_files / _EXEC_ID)


async def test_an_autopilot_call_keeps_its_copies_in_its_own_folder(
    exec_files: Path,
) -> None:
    first, second = _SAME_NAME["workspace"]

    with separate_exec_files(_EXEC_ID):
        call_folder = Path(get_exec_file_path(_EXEC_ID, ""))
        copies = [await _stage(first), await _stage(second)]

        assert [copy.parent.parent for copy in copies] == [call_folder] * 2
        assert [copy.read_bytes() for copy in copies] == [
            _CONTENT[first],
            _CONTENT[second],
        ]

    assert list(exec_files.iterdir()) == []


@pytest.mark.parametrize(
    "base_64, file_out, stored",
    [
        (
            False,
            "workspace://stored#text/csv",
            [
                call(
                    content=_CONTENT[_REPORT],
                    filename="report.csv",
                    rename_on_conflict=True,
                )
            ],
        ),
        (
            True,
            f"data:text/csv;base64,{base64.b64encode(_CONTENT[_REPORT]).decode()}",
            [],
        ),
    ],
    ids=["for_block_output", "for_external_api"],
)
async def test_store_file_stores_a_staged_copy_handed_to_it(
    base_64: bool, file_out: str, stored: list, workspace: MagicMock
) -> None:
    """Store File hands its input, here the path a staging returned, back to
    ``store_media_file``: as ``for_block_output``, or as ``for_external_api``
    with ``base_64`` on. The file goes out under its own name."""
    local_path = await store_media_file(
        MediaFileType(_REPORT), _CONTEXT, return_format="for_local_processing"
    )

    block = FileStoreBlock()
    outputs = [
        output
        async for output in block.run(
            block.Input(file_in=local_path, base_64=base_64),
            execution_context=_CONTEXT,
        )
    ]

    assert outputs == [("file_out", file_out)]
    assert workspace.write_file.await_args_list == stored


async def test_a_staged_path_handed_back_names_the_same_copy() -> None:
    local_path = await store_media_file(
        MediaFileType(_REPORT), _CONTEXT, return_format="for_local_processing"
    )

    again = await store_media_file(
        local_path, _CONTEXT, return_format="for_local_processing"
    )

    assert again == local_path
    assert Path(get_exec_file_path(_EXEC_ID, again)).read_bytes() == _CONTENT[_REPORT]


@pytest.mark.parametrize(
    "local_path",
    [
        "../outside.txt",
        "{copy_folder}/../../outside.txt",
        "{outside}",
        "{copy_folder}/link/secret.txt",
    ],
    ids=["parent", "up-from-a-copy", "absolute", "symlink"],
)
async def test_a_local_path_out_of_the_run_folder_is_refused(
    local_path: str, exec_files: Path, tmp_path: Path
) -> None:
    """Each path names a real file outside the run's folder, so a broken
    containment check would read it rather than fail to find it."""
    copy = await _stage(_REPORT)
    outside = exec_files / "outside.txt"
    outside.write_bytes(b"not this run's")
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "secret.txt").write_bytes(b"not this run's")
    (copy.parent / "link").symlink_to(secrets)

    with pytest.raises(ValueError):
        await store_media_file(
            MediaFileType(
                local_path.format(copy_folder=copy.parent.name, outside=outside)
            ),
            _CONTEXT,
            return_format="for_external_api",
        )


async def _stage(reference: str) -> Path:
    """Stage a file for local processing as a block does, and return where
    its copy is."""
    local_path = await store_media_file(
        MediaFileType(reference), _CONTEXT, return_format="for_local_processing"
    )
    return Path(get_exec_file_path(_EXEC_ID, local_path))
