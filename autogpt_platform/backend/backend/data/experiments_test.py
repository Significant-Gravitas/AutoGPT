"""Server-side experiment assignment."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import posthog.client
import pytest
import urllib3.connectionpool
from posthog import Posthog

import backend.util.feature_flag.posthog as ph
from backend.data import experiments
from backend.data.experiments import Experiment
from backend.util.feature_flag import evaluate_experiment_variant

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


async def test_the_arm_is_read_locally_and_reported_as_an_exposure(
    mocker, posthog_client
):
    """`$feature_flag_called` is how the experiment counts who was shown an arm."""
    capture = mocker.spy(posthog_client, "capture")
    remote = _remote_flags_request(mocker)

    arm = await evaluate_experiment_variant(PRICING, "u-1")

    assert arm in ARMS
    assert [call.args[0] for call in capture.call_args_list] == ["$feature_flag_called"]
    remote.assert_not_called()


async def test_without_local_definitions_nobody_is_enrolled_rather_than_asking_posthog(
    mocker, posthog_client
):
    posthog_client.feature_flags = []
    remote = _remote_flags_request(mocker)

    assert await evaluate_experiment_variant(PRICING, "u-1") is None
    remote.assert_not_called()


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
