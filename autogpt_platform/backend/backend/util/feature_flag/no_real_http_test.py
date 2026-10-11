"""The flag package's network guard refuses what it exists to refuse."""

import pytest
import urllib3
from posthog import Posthog


def test_a_request_off_the_machine_is_refused_and_recorded(no_real_http):
    # A host that never resolves, so a guard that stopped applying fails here
    # without reaching anything real.
    with pytest.raises(urllib3.exceptions.ProtocolError):
        urllib3.PoolManager().request("GET", "https://example.invalid/flags?token=x")

    assert no_real_http == ["GET https://example.invalid/flags"]
    no_real_http.clear()


def test_a_captured_event_is_refused_and_recorded(no_real_http):
    client = Posthog("phc_test", sync_mode=True)
    try:
        client.capture("$feature_flag_called", distinct_id="u-1")
    finally:
        client.shutdown()

    assert no_real_http == ["a $feature_flag_called event"]
    no_real_http.clear()
