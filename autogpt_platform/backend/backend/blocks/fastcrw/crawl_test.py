from unittest.mock import MagicMock

from backend.blocks.fastcrw._api import ScrapeFormat
from backend.blocks.fastcrw._config import fastcrw
from backend.blocks.fastcrw.crawl import FastCRWCrawlBlock


def _option_value(options, key: str):
    if hasattr(options, key):
        return getattr(options, key)
    return options[key]


def test_crawl_clamps_negative_provider_params():
    block = FastCRWCrawlBlock()
    app = MagicMock()
    result = object()
    app.crawl.return_value = result

    input_data = block.Input(
        credentials=fastcrw.get_test_credentials().model_dump(),
        url="https://example.com",
        limit=-5,
        max_age=-1,
        wait_for=-10,
        formats=[ScrapeFormat.MARKDOWN],
    )

    assert block._crawl(app, input_data) is result

    app.crawl.assert_called_once()
    args, kwargs = app.crawl.call_args
    assert args == ("https://example.com",)
    assert kwargs["limit"] == 0

    scrape_options = kwargs["scrape_options"]
    assert _option_value(scrape_options, "max_age") == 0
    assert _option_value(scrape_options, "wait_for") == 0
