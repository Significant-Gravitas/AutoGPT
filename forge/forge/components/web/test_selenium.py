from pathlib import Path

import pytest

from forge.llm.providers.multi import MultiProvider

from . import BrowsingError, WebSeleniumComponent


@pytest.fixture
def web_selenium_component(app_data_dir: Path):
    return WebSeleniumComponent(MultiProvider(), app_data_dir)


@pytest.mark.asyncio
async def test_browse_website_nonexistent_url(
    web_selenium_component: WebSeleniumComponent,
):
    url = "https://auto-gpt-thinks-this-website-does-not-exist.com"
    question = "How to execute a barrel roll"

    with pytest.raises(BrowsingError, match="NAME_NOT_RESOLVED") as raised:
        await web_selenium_component.read_webpage(url=url, question=question)

        # Sanity check that the response is not too long
        assert len(raised.exconly()) < 200
