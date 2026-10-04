import pytest

from backend.blocks._base import Block, BlockEffect
from backend.blocks.google_maps_directions import GoogleMapsGetDirectionsBlock
from backend.blocks.google_maps_places import (
    GoogleMapsResolveLinksBlock,
    GoogleMapsResolvePlacesBlock,
)
from backend.blocks.google_maps_weather import GoogleMapsWeatherBlock


@pytest.mark.parametrize(
    ("block", "effect"),
    [
        (GoogleMapsGetDirectionsBlock, BlockEffect.READ),
        (GoogleMapsResolvePlacesBlock, BlockEffect.READ),
        (GoogleMapsResolveLinksBlock, BlockEffect.READ),
        (GoogleMapsWeatherBlock, BlockEffect.READ),
    ],
)
def test_block_declares_what_it_does(block: type[Block], effect: BlockEffect):
    # AutoPilot runs reads and workspace saves without asking, asks before
    # anything that changes Google, and treats an undeclared block as the latter.
    assert block().effect is effect
