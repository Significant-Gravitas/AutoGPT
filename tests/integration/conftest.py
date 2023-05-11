import os

import openai
import pytest

from tests.conftest import PROXY
from tests.vcr.vcr_filter import before_record_request, before_record_response


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
        "match_on": ["method", "body"],
    }


def patch_api_base(requestor):
    new_api_base = f"{PROXY}/v1"
    requestor.api_base = new_api_base
    return requestor


@pytest.fixture
def patched_api_requestor(mocker):
    original_init = openai.api_requestor.APIRequestor.__init__

    def patched_init(requestor, *args, **kwargs):
        original_init(requestor, *args, **kwargs)
        patch_api_base(requestor)

    if PROXY:
        mocker.patch("openai.api_requestor.APIRequestor.__init__", new=patched_init)

    return mocker
