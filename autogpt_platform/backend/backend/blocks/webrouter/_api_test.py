"""Unit tests for the keyless router the WebRouter blocks build: which
providers it may call, whatever API keys this server's environment holds."""

import pytest
from web_router import WebRouterError

from backend.blocks.webrouter._api import KEY_VARIABLES, Method, keyless_router

PLATFORM_KEYS = ("TAVILY_API_KEY", "EXA_API_KEY", "JINA_API_KEY", "PARALLEL_API_KEY")


@pytest.fixture(autouse=True)
def no_provider_keys(monkeypatch: pytest.MonkeyPatch):
    for var in (*PLATFORM_KEYS, *KEY_VARIABLES.values()):
        monkeypatch.delenv(var, raising=False)


async def members(method: Method) -> tuple[str, ...]:
    async with keyless_router(method) as router:
        if method == "search":
            return router.get_provider("auto").members
        return router.get_extract_provider("auto").members


@pytest.mark.asyncio
async def test_platform_keys_never_reach_the_chain(monkeypatch: pytest.MonkeyPatch):
    for var in PLATFORM_KEYS:
        monkeypatch.setenv(var, "platform-key")

    assert await members("search") == (
        "nimble",
        "you-mcp",
        "exa-mcp",
        "parallel-mcp",
        "keenable",
    )
    assert await members("extract") == ("nimble", "firecrawl", "keenable")


@pytest.mark.asyncio
async def test_a_member_whose_own_key_is_set_is_left_out(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NIMBLE_API_KEY", "operator-key")
    monkeypatch.setenv("KEENABLE_API_KEY", "operator-key")

    assert await members("search") == ("you-mcp", "exa-mcp", "parallel-mcp")
    assert await members("extract") == ("firecrawl",)


@pytest.mark.asyncio
async def test_a_blank_key_does_not_remove_its_member(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NIMBLE_API_KEY", "  ")
    assert (await members("extract"))[0] == "nimble"


def test_no_keyless_member_left_is_an_error(monkeypatch: pytest.MonkeyPatch):
    for var in KEY_VARIABLES.values():
        monkeypatch.setenv(var, "operator-key")

    with pytest.raises(WebRouterError, match="No keyless extract provider"):
        keyless_router("extract")
