import os

import openai.api_requestor
import pytest
from pytest_mock import MockerFixture

from .vcr_filter import PROXY, before_record_request, before_record_response

DEFAULT_RECORD_MODE = "new_episodes"
BASE_VCR_CONFIG = {
    "before_record_request": before_record_request,
    "before_record_response": before_record_response,
    "filter_headers": [
        "Authorization",
        "X-OpenAI-Client-User-Agent",
        "User-Agent",
    ],
    "match_on": ["method", "body"],
}


@pytest.fixture(scope="session")
def vcr_config(get_base_vcr_config):
    return get_base_vcr_config


@pytest.fixture(scope="session")
def get_base_vcr_config(request):
    record_mode = request.config.getoption("--record-mode", default="new_episodes")
    config = BASE_VCR_CONFIG

    if record_mode is None:
        config["record_mode"] = DEFAULT_RECORD_MODE

    return config


@pytest.fixture()
def vcr_cassette_dir(request):
    test_name = os.path.splitext(request.node.name)[0]
    return os.path.join("tests/Auto-GPT-test-cassettes", test_name)


def patch_api_base(requestor):
    new_api_base = f"{PROXY}/v1"
    requestor.api_base = new_api_base
    return requestor


@pytest.fixture
def patched_api_requestor(mocker: MockerFixture):
    original_init = openai.api_requestor.APIRequestor.__init__
    original_validate_headers = openai.api_requestor.APIRequestor._validate_headers

    def patched_init(requestor, *args, **kwargs):
        original_init(requestor, *args, **kwargs)
        patch_api_base(requestor)

    def patched_validate_headers(self, supplied_headers):
        headers = original_validate_headers(self, supplied_headers)
        headers["AGENT-MODE"] = os.environ.get("AGENT_MODE")
        headers["AGENT-TYPE"] = os.environ.get("AGENT_TYPE")
        return headers

    if PROXY:
        mocker.patch("openai.api_requestor.APIRequestor.__init__", new=patched_init)
        mocker.patch.object(
            openai.api_requestor.APIRequestor,
            "_validate_headers",
            new=patched_validate_headers,
        )
