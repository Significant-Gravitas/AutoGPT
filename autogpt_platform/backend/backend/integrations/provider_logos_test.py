from pathlib import Path

import pytest
from PIL import Image

from backend.api.features.integrations.models import get_all_provider_names
from backend.blocks import load_all_blocks
from backend.integrations.mcp_catalog import MCPCatalogEntry, get_mcp_catalog

LOGO_DIRECTORY = (
    Path(__file__).resolve().parents[3] / "frontend" / "public" / "integrations"
)


def block_provider_names() -> list[str]:
    load_all_blocks()
    return get_all_provider_names()


@pytest.mark.parametrize("provider", block_provider_names())
def test_every_block_provider_has_a_logo(provider: str):
    # Match the display aliases in toConnectableProviders and integrationIconSrc.
    aliases = {"codex": "openai.png", "microsoft_365_copilot": "microsoft.webp"}
    assert_logo_image(aliases.get(provider, f"{provider}.png"))


@pytest.mark.parametrize("entry", get_mcp_catalog(), ids=lambda entry: entry.name)
def test_every_mcp_provider_has_a_logo(entry: MCPCatalogEntry):
    assert entry.mcp_server.icon_id, f"{entry.name} must declare an icon_id"
    assert_logo_image(f"{entry.mcp_server.icon_id}.png")


def assert_logo_image(filename: str):
    path = LOGO_DIRECTORY / filename
    assert path.is_file(), f"Missing provider logo: public/integrations/{filename}"
    with Image.open(path) as image:
        image.load()
        assert image.width > 0 and image.height > 0, f"Empty provider logo: {filename}"
