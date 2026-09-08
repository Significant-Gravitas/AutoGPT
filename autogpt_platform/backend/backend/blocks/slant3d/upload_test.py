from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend.blocks.slant3d._api import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.slant3d.slicing import Slant3DSlicerBlock
from backend.data.execution import ExecutionContext


@pytest.mark.parametrize("read_error", [None, PermissionError("Access denied")])
async def test_slicer_loads_workspace_attachment_with_user_context(
    tmp_path, read_error
):
    context = ExecutionContext(
        user_id="user-1",
        graph_exec_id="run-1",
        workspace_id="workspace-1",
        session_id="session-1",
    )
    block = Slant3DSlicerBlock()
    workspace = Mock()
    workspace.read_file_by_id = AsyncMock(
        return_value=b"STL bytes", side_effect=read_error
    )
    workspace.get_file_info = AsyncMock(return_value=Mock(name="file-info"))
    workspace.get_file_info.return_value.name = "CalibrationCube.stl"
    with (
        patch("backend.util.file.TEMP_DIR", tmp_path),
        patch("backend.util.file.get_cloud_storage_handler", AsyncMock()),
        patch(
            "backend.util.workspace.WorkspaceManager", return_value=workspace
        ) as manager,
        patch("backend.util.file.scan_content_safe", AsyncMock()) as scan,
        patch("backend.blocks.slant3d.base.Requests") as requests,
        patch.object(
            block,
            "_make_request",
            AsyncMock(
                side_effect=[
                    {
                        "data": {
                            "presignedUrl": "https://upload.example.com/model",
                            "filePlaceholder": {"id": "placeholder"},
                        }
                    },
                    {"data": {"publicFileServiceId": "file-1"}},
                    {"message": "File Price Estimated", "data": {"total": 1.37}},
                ]
            ),
        ) as api,
    ):
        requests.return_value.get = AsyncMock(
            side_effect=ValueError("URL scheme 'workspace' is not allowed")
        )
        requests.return_value.put = AsyncMock()
        outputs = block.run(
            block.Input(
                credentials=TEST_CREDENTIALS_INPUT,
                file_url="workspace://attachment-1",
                platform_id="platform-1",
            ),
            credentials=TEST_CREDENTIALS,
            execution_context=context,
        )
        if read_error:
            with pytest.raises(PermissionError, match="Access denied"):
                _ = [item async for item in outputs]
            api.assert_not_awaited()
            requests.return_value.put.assert_not_awaited()
            return
        result = dict([item async for item in outputs])
    manager.assert_called_once_with("user-1", "workspace-1", "session-1", scope=None)
    workspace.read_file_by_id.assert_awaited_once_with("attachment-1")
    scan.assert_awaited_once_with(b"STL bytes", filename="CalibrationCube.stl")
    requests.return_value.get.assert_not_awaited()
    requests.return_value.put.assert_awaited_once_with(
        "https://upload.example.com/model",
        data=b"STL bytes",
        headers={"Content-Type": "application/octet-stream"},
    )
    assert api.await_args_list[0].kwargs["json"] == {
        "name": "CalibrationCube.stl",
        "platformId": "platform-1",
    }
    assert result["price"] == 1.37
