import logging
import os
from hashlib import sha256

import pytest
from openai import OpenAI
from openai._models import FinalRequestOptions
from openai._types import Omit
from openai._utils import is_given
from pytest_mock import MockerFixture

from .vcr_filter import (
    before_record_request,
    before_record_response,
    freeze_request_body,
)

DEFAULT_RECORD_MODE = "new_episodes"
BASE_VCR_CONFIG = {
    "before_record_request": before_record_request,
    "before_record_response": before_record_response,
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
    return os.path.join("tests/vcr_cassettes", test_name)


@pytest.fixture
def cached_openai_client(mocker: MockerFixture) -> OpenAI:
    client = OpenAI()
    _prepare_options = client._prepare_options

    def _patched_prepare_options(self, options: FinalRequestOptions):
        _prepare_options(options)

        headers: dict[str, str | Omit] = (
            {**options.headers} if is_given(options.headers) else {}
        )
        options.headers = headers
        data: dict = options.json_data

        logging.getLogger("cached_openai_client").debug(
            f"Outgoing API request: {headers}\n{data if data else None}"
        )

        # Add hash header for cheap & fast matching on cassette playback
        headers["X-Content-Hash"] = sha256(
            freeze_request_body(data), usedforsecurity=False
        ).hexdigest()

    mocker.patch.object(
        client,
        "_prepare_options",
        new=_patched_prepare_options,
    )

    return client
