import pytest

from tests.vcr.openai_filter import before_record_request, before_record_response


@pytest.fixture(scope="session")
def vcr_config():
    # this fixture is called by the pytest-recording vcr decorator.
    return {
        "record_mode": "new_episodes",
        "before_record_request": before_record_request,
        "before_record_response": before_record_response,
        "filter_headers": [
            "Authorization",
            "X-OpenAI-Client-User-Agent",
            "User-Agent",
        ],
    }
