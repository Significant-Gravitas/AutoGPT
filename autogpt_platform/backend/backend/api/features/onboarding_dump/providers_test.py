"""Tests for the shared provider registry both LLM jobs are given.

The recommender picks from this list and the greeting promises
automations built on it, so the graceful-degradation guarantee — a
registry that still answers when block loading fails — is what keeps a
bad import from turning either job into a wrong answer.
"""

from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump import providers

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
