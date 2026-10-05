import pytest

from backend.blocks._base import Block, BlockEffect
from backend.blocks.google.analytics import (
    GoogleAnalyticsListDimensionsAndMetricsBlock,
    GoogleAnalyticsListPropertiesBlock,
)
from backend.blocks.google.analytics_realtime import (
    GoogleAnalyticsRunRealtimeReportBlock,
)
from backend.blocks.google.analytics_reports import GoogleAnalyticsRunReportBlock


@pytest.mark.parametrize(
    ("block", "effect"),
    [
        (GoogleAnalyticsListPropertiesBlock, BlockEffect.READ),
        (GoogleAnalyticsListDimensionsAndMetricsBlock, BlockEffect.READ),
        (GoogleAnalyticsRunReportBlock, BlockEffect.READ),
        (GoogleAnalyticsRunRealtimeReportBlock, BlockEffect.READ),
    ],
)
def test_block_declares_what_it_does(block: type[Block], effect: BlockEffect):
    # AutoPilot runs reads and workspace saves without asking, asks before
    # anything that changes Google, and treats an undeclared block as the latter.
    assert block().effect is effect
