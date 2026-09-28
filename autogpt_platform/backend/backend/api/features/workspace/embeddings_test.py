import asyncio
from unittest.mock import AsyncMock

import pytest

from backend.api.features.workspace import embeddings


@pytest.fixture
def embedding_calls(mocker):
    embeddings._embedded.clear()
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
    embeddings._embedded.clear()


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


async def test_one_text_is_requested_once_and_other_text_does_not_wait(
    embedding_calls,
):
    generate, _ = embedding_calls
    started, release = asyncio.Event(), asyncio.Event()

    async def respond(text: str) -> list[float]:
        if text == "slow":
            started.set()
            await release.wait()
        return [float(len(text))]

    generate.side_effect = respond
    first = asyncio.create_task(embeddings._embed("m", "slow"))
    second = asyncio.create_task(embeddings._embed("m", "slow"))
    await started.wait()

    # Kills: one lock across every text, which parks this behind "slow".
    assert await asyncio.wait_for(embeddings._embed("m", "fast"), timeout=2) == [4.0]
    release.set()
    assert await first == await second == [4.0]
    # Kills: a second request for text that is already in flight.
    assert sorted(c.args[0] for c in generate.await_args_list) == ["fast", "slow"]


async def test_a_failed_request_is_retried_rather_than_cached(embedding_calls):
    generate, _ = embedding_calls
    generate.side_effect = [RuntimeError("rate limited"), [1.0]]

    with pytest.raises(RuntimeError):
        await embeddings._embed("m", "text")
    # Kills: remembering the failure for a day.
    assert await embeddings._embed("m", "text") == [1.0]
