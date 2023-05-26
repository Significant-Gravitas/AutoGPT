import os

import openai.api_requestor
import pytest
from pytest_mock import MockerFixture

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


import numpy
import pytest
from pytest_mock import MockerFixture

import autogpt.memory.vector.memory_item as vector_memory_item
import autogpt.memory.vector.providers.base as memory_provider_base
from autogpt.config.config import Config
from autogpt.llm.providers.openai import OPEN_AI_EMBEDDING_MODELS
from autogpt.memory.vector import get_memory
from autogpt.memory.vector.utils import Embedding


@pytest.fixture
def embedding_dimension(config: Config):
    return OPEN_AI_EMBEDDING_MODELS[config.embedding_model].embedding_dimensions


@pytest.fixture
def mock_embedding(embedding_dimension: int) -> Embedding:
    return numpy.full((1, embedding_dimension), 0.0255, numpy.float32)[0]


@pytest.fixture
def mock_get_embedding(mocker: MockerFixture, embedding_dimension: int):
    mocker.patch.object(
        vector_memory_item,
        "get_embedding",
        return_value=[0.0255] * embedding_dimension,
    )
    mocker.patch.object(
        memory_provider_base,
        "get_embedding",
        return_value=[0.0255] * embedding_dimension,
    )


@pytest.fixture
def memory_none(agent_test_config: Config, mock_get_embedding):
    was_memory_backend = agent_test_config.memory_backend

    agent_test_config.set_memory_backend("no_memory")
    yield get_memory(agent_test_config)

    agent_test_config.set_memory_backend(was_memory_backend)
