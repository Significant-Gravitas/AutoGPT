from typing import Callable

import pytest
from ldclient import LDClient
from pytest_mock import MockerFixture

import backend.util.feature_flag as feature_flag


@pytest.fixture
def serve_launchdarkly(mocker: MockerFixture) -> Callable[[LDClient], None]:
    """Answer flag reads from a given client, on a deployment with an SDK key.

    Without a key a read never asks LaunchDarkly at all, so a client standing
    in for a live one needs the key that a live one would have.
    """

    def serve(client: LDClient) -> None:
        mocker.patch.object(
            feature_flag.settings.secrets, "launch_darkly_sdk_key", "sdk-test"
        )
        mocker.patch("backend.util.feature_flag.ldclient.get", return_value=client)

    return serve
