import pytest

from tests.vcr.openai_filter import before_record_request


@pytest.fixture
def vcr_config():
    # this fixture is called by the pytest-recording vcr decorator.
    return {
        "record_mode": "new_episodes",
        "before_record_request": before_record_request,
        "filter_headers": [
            "authorization",
            "X-OpenAI-Client-User-Agent",
            "User-Agent",
        ],
    }
