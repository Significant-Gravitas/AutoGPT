"""Server-side experiment assignment."""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import posthog.client
import pytest
import urllib3.connectionpool
from posthog import Posthog

import backend.util.feature_flag as ff
import backend.util.feature_flag.posthog as ph
from backend.data import experiments
from backend.data.experiments import Experiment
from backend.util.feature_flag import (
    evaluate_experiment_variant,
    get_feature_flag_value,
)
from backend.util.settings import FeatureFlagBackend

PRICING = Experiment.SUBSCRIPTION_PRICING.value
ARMS = ("control", "yearly-pro")


@pytest.fixture
def posthog_client(mocker, monkeypatch):
    """A real PostHog client holding the experiment's definition, with no network."""
    monkeypatch.setattr(
        urllib3.connectionpool.HTTPConnectionPool, "urlopen", _refuse_network
    )
    client = Posthog(
        "phc_test",
        host="http://127.0.0.1:9",
        secret_key="phx_test",
        enable_local_evaluation=False,
        send=False,
        sync_mode=True,
    )
    client.feature_flags = [
        {
            "id": 1,
            "key": PRICING,
            "active": True,
            "filters": {
                "groups": [{"properties": [], "rollout_percentage": 100}],
                "multivariate": {
                    "variants": [{"key": arm, "rollout_percentage": 50} for arm in ARMS]
                },
            },
        }
    ]
    mocker.patch.object(ph, "get_flag_client", return_value=client)
    return client


async def test_the_arm_is_read_locally_and_tells_posthog_nothing(
    mocker, posthog_client
):
    capture = mocker.spy(posthog_client, "capture")
    remote = _remote_flags_request(mocker)

    arm = await evaluate_experiment_variant(PRICING, "u-1")

    assert arm in ARMS
    capture.assert_not_called()
    remote.assert_not_called()


async def test_no_server_read_of_an_experiment_reports_its_exposure(
    mocker, posthog_client
):
    """The browser reports it, consent-gated; a server event would double it and
    skip consent, whichever read made it."""
    mocker.patch.object(
        ff.settings.config, "feature_flag_backend", FeatureFlagBackend.POSTHOG
    )
    capture = mocker.spy(posthog_client, "capture")
    _remote_flags_request(mocker)

    for experiment in Experiment:
        await get_feature_flag_value(experiment.value, "u-1")
        await evaluate_experiment_variant(experiment.value, "u-1")

    capture.assert_not_called()


@pytest.mark.parametrize("definitions", [[], None], ids=["loaded-empty", "not-loaded"])
async def test_without_local_definitions_nobody_is_enrolled_rather_than_asking_posthog(
    mocker, posthog_client, definitions
):
    """Nor does the read wait on a definitions fetch, which the SDK starts itself
    when it has never loaded any."""
    # The public setter turns None into []; None is the SDK's state before a load.
    posthog_client._feature_flags = definitions
    load = mocker.spy(posthog_client, "load_feature_flags")
    remote = _remote_flags_request(mocker)

    assert await evaluate_experiment_variant(PRICING, "u-1") is None
    load.assert_not_called()
    remote.assert_not_called()


async def test_a_failed_user_lookup_enrols_nobody(mocker, posthog_client):
    """The first arm is stored for good, so it is never read without the user's
    properties: an email- or role-targeted arm would be missed."""
    mocker.patch.object(
        ff, "_fetch_user_context", AsyncMock(side_effect=ConnectionError("db down"))
    )

    assert await evaluate_experiment_variant(PRICING, str(uuid.uuid4())) is None


async def test_a_stored_arm_wins_over_posthogs_current_bucket(mocker):
    _stored_row(mocker, variant="control")
    evaluate = mocker.patch.object(
        experiments, "evaluate_experiment_variant", AsyncMock(return_value="yearly-pro")
    )

    assignment = await experiments.assign("u-1", Experiment.SUBSCRIPTION_PRICING)

    assert assignment is not None and assignment.variant == "control"
    evaluate.assert_not_awaited()


@pytest.mark.parametrize("arm", ["yearly-pro", None])
async def test_the_first_read_stores_posthogs_arm_and_nothing_when_not_enrolled(
    mocker, arm
):
    _stored_row(mocker, variant=None)
    mocker.patch.object(
        experiments, "evaluate_experiment_variant", AsyncMock(return_value=arm)
    )
    record = mocker.patch.object(experiments, "record_assignment", AsyncMock())

    await experiments.assign("u-1", Experiment.SUBSCRIPTION_PRICING)

    if arm is None:
        record.assert_not_awaited()
    else:
        record.assert_awaited_once_with("u-1", PRICING, arm, source="backend")


def _stored_row(mocker, variant: str | None) -> None:
    row = None
    if variant is not None:
        row = mocker.Mock(
            experimentKey=PRICING,
            variant=variant,
            source="backend",
            createdAt=datetime(2026, 10, 11, tzinfo=UTC),
        )
    table = mocker.patch("prisma.models.ExperimentAssignment.prisma")
    table.return_value.find_unique = AsyncMock(return_value=row)


def _refuse_network(pool, method, url, *args, **kwargs):
    raise urllib3.exceptions.ProtocolError(f"no network in tests: {method} {url}")


def _remote_flags_request(mocker):
    """The SDK's /flags call: a read that reaches it has told PostHog who the user is."""
    return mocker.patch.object(posthog.client, "flags", return_value={"flags": {}})
