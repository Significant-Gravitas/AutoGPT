from unittest.mock import AsyncMock

import pytest

from backend.api.features.workspace import embeddings


@pytest.fixture
def embedding_calls(mocker):
    embeddings._embed.cache_clear()
    mocker.patch.object(
        embeddings, "get_content_embedding", AsyncMock(return_value=None)
    )
    store = mocker.patch.object(
        embeddings, "store_content_embedding", AsyncMock(return_value=True)
    )
    generate = mocker.patch.object(
        embeddings, "generate_embedding", AsyncMock(return_value=[0.5, 0.25])
    )
    yield generate, store
    embeddings._embed.cache_clear()


async def test_files_with_the_same_text_share_one_embedding_call(embedding_calls):
    generate, store = embedding_calls

    # Three hires' copies of one skill, then a file with other text.
    for i in range(3):
        await embeddings._run_embedding(
            f"file-{i}", f"user-{i}", "SKILL.md", f"/experts/e{i}/skills/s/SKILL.md"
        )
    await embeddings._run_embedding("file-9", "user-9", "notes.md", "/notes.md")

    # Kills: embedding each file through its own OpenAI call.
    assert generate.await_count == 2
    assert [c.kwargs["content_id"] for c in store.await_args_list] == [
        "file-0",
        "file-1",
        "file-2",
        "file-9",
    ]
    assert all(c.kwargs["embedding"] == [0.5, 0.25] for c in store.await_args_list)
    assert store.await_args_list[0].kwargs["searchable_text"] == "SKILL.md SKILL"
