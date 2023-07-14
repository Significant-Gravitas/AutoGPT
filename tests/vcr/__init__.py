import os
from hashlib import sha256

import openai.api_requestor
import pytest
from pytest_mock import MockerFixture

from .vcr_filter import (
    PROXY,
    before_record_request,
    before_record_response,
    freeze_request_body,
)

DEFAULT_RECORD_MODE = "new_episodes"
BASE_VCR_CONFIG = {
    "before_record_request": before_record_request,
    "before_record_response": before_record_response,
    "filter_headers": [
        "Authorization",
        "AGENT-MODE",
        "AGENT-TYPE",
        "OpenAI-Organization",
        "X-OpenAI-Client-User-Agent",
        "User-Agent",
    ],
    "match_on": ["method", "headers"],
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


def patch_api_base(requestor: openai.api_requestor.APIRequestor):
    new_api_base = f"{PROXY}/v1"
    requestor.api_base = new_api_base
    return requestor


@pytest.fixture
def patched_api_requestor(mocker: MockerFixture):
    init_requestor = openai.api_requestor.APIRequestor.__init__
    prepare_request = openai.api_requestor.APIRequestor._prepare_request_raw

    def patched_init_requestor(requestor, *args, **kwargs):
        init_requestor(requestor, *args, **kwargs)
        patch_api_base(requestor)

    def patched_prepare_request(self, *args, **kwargs):
        url, headers, data = prepare_request(self, *args, **kwargs)

        if PROXY:
            headers["AGENT-MODE"] = os.environ.get("AGENT_MODE")
            headers["AGENT-TYPE"] = os.environ.get("AGENT_TYPE")

        # Add hash header for cheap & fast matching on cassette playback
        headers["X-Content-Hash"] = sha256(
            freeze_request_body(data), usedforsecurity=False
        ).hexdigest()

        return url, headers, data

    if PROXY:
        mocker.patch.object(
            openai.api_requestor.APIRequestor,
            "__init__",
            new=patched_init_requestor,
        )
    mocker.patch.object(
        openai.api_requestor.APIRequestor,
        "_prepare_request_raw",
        new=patched_prepare_request,
    )
