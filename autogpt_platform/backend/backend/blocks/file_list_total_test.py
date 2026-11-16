"""Blocks that load a list of files refuse it once the files together pass the
file limit, before loading the next one."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.ai_image_customizer import TEST_CREDENTIALS as REPLICATE_CREDENTIALS
from backend.blocks.ai_image_customizer import (
    TEST_CREDENTIALS_INPUT as REPLICATE_CREDENTIALS_INPUT,
)
from backend.blocks.ai_image_customizer import AIImageCustomizerBlock
from backend.blocks.github._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.github.commits import GithubMultiFileCommitBlock
from backend.data.execution import ExecutionContext

_DATA_URI = "data:image/png;base64," + "A" * 18  # 40 characters


async def _outputs(block, input_data, credentials) -> dict:
    return {
        name: value
        async for name, value in block.run(
            input_data,
            credentials=credentials,
            execution_context=ExecutionContext(user_id="u1", graph_exec_id="g1"),
        )
    }


@pytest.mark.asyncio
async def test_github_multi_file_commit_limits_the_total():
    block = GithubMultiFileCommitBlock()
    input_data = GithubMultiFileCommitBlock.Input.model_validate(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "repo_url": "https://github.com/owner/repo",
            "branch": "main",
            "commit_message": "files",
            "files": [
                {"path": f"f{i}.png", "content": "workspace://a", "operation": "upsert"}
                for i in range(50)
            ],
        }
    )
    resolve = AsyncMock(return_value=_DATA_URI)
    commit = AsyncMock(return_value=("sha", "url"))

    with (
        patch("backend.blocks.github.commits.resolve_media_content", resolve),
        patch.object(block, "multi_file_commit", commit),
        patch("backend.util.file.MAX_TOTAL_FILE_SIZE_BYTES", 100),
    ):
        outputs = await _outputs(block, input_data, TEST_CREDENTIALS)

    assert "Files too large" in outputs["error"]
    assert resolve.await_count == 3
    commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_ai_image_customizer_limits_the_total():
    block = AIImageCustomizerBlock()
    input_data = AIImageCustomizerBlock.Input.model_validate(
        {
            "credentials": REPLICATE_CREDENTIALS_INPUT,
            "prompt": "make it blue",
            "images": ["workspace://a"] * 50,
        }
    )
    store = AsyncMock(return_value=_DATA_URI)
    run_model = AsyncMock()

    with (
        patch("backend.blocks.ai_image_customizer.store_media_file", store),
        patch.object(block, "run_model", run_model),
        patch("backend.util.file.MAX_TOTAL_FILE_SIZE_BYTES", 100),
    ):
        outputs = await _outputs(block, input_data, REPLICATE_CREDENTIALS)

    assert "Files too large" in outputs["error"]
    assert store.await_count == 3
    run_model.assert_not_awaited()
