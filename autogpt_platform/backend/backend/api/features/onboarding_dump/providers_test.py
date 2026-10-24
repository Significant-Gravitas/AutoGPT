"""Tests for the shared provider registry both LLM jobs are given.

The recommender picks from this list and the greeting promises
automations built on it, so the graceful-degradation guarantee — a
registry that still answers when block loading fails — is what keeps a
bad import from turning either job into a wrong answer.
"""

import pytest
from pytest_mock import MockerFixture

from backend.api.features.integrations import support
from backend.api.features.integrations.support import ConnectableService
from backend.api.features.onboarding_dump import providers
from backend.api.features.onboarding_dump.models import SuggestedPrompt

MODULE = "backend.api.features.onboarding_dump.providers"


def test_known_providers_pairs_each_id_with_its_description(mocker: MockerFixture):
    mocker.patch(f"{MODULE}.load_all_blocks")
    mocker.patch(f"{MODULE}.get_all_provider_names", return_value=["slack", "notion"])
    mocker.patch(
        f"{MODULE}.get_provider_description",
        side_effect={"slack": "Team chat", "notion": None}.get,
    )

    assert providers.known_providers() == {"slack": "Team chat", "notion": None}


def test_known_providers_still_answers_when_block_loading_fails(mocker: MockerFixture):
    # Statically registered providers are already on the registry, so a
    # broken block import costs the SDK-registered ones — not the list.
    mocker.patch(f"{MODULE}.load_all_blocks", side_effect=RuntimeError("bad import"))
    mocker.patch(f"{MODULE}.get_all_provider_names", return_value=["slack"])
    mocker.patch(f"{MODULE}.get_provider_description", return_value="Team chat")

    assert providers.known_providers() == {"slack": "Team chat"}


def test_provider_lines_omits_the_colon_for_undescribed_providers():
    lines = providers.provider_lines({"slack": "Team chat", "notion": None})

    assert lines == "- slack: Team chat\n- notion"


def test_provider_lines_is_empty_for_an_empty_registry():
    # The greeting prompt drops the whole integrations block on this, so
    # an empty string here is the difference between no constraint and a
    # constraint that names nothing.
    assert providers.provider_lines({}) == ""


@pytest.mark.parametrize(
    "integrations, expected",
    [
        ([], ["Shopify"]),  # the model said "no sign-in"
        (["printify"], ["Shopify"]),  # the model left Shopify out
        (["Shopify store"], ["Shopify"]),  # named twice, labelled once
        (None, ["Shopify"]),  # a row stored before the field
    ],
)
def test_a_task_naming_a_tool_we_lack_is_labelled_whatever_it_declares(
    mocker: MockerFixture, integrations: list[str] | None, expected: list[str]
):
    """The model's integrations list is a claim, not the check: a Shopify task
    that declares none (or the wrong ones) still says it needs Shopify."""
    mocker.patch.object(
        support,
        "connectable_services",
        return_value=[ConnectableService(key="printify", name="Printify")],
    )
    task = SuggestedPrompt(
        title="Sync new Printify products to your Shopify store",
        prompt="Each night, push new designs to Shopify.",
        integrations=integrations,
    )

    [labelled] = providers.label_support([task])

    assert labelled.unsupported == expected
