"""The server-owned description of an AI connection."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_mock

from backend.copilot import offers, provider_tiers, transports
from backend.copilot.offers import (
    EntitlementUnavailable,
    advanced_tier_allowed,
    advanced_tier_entitled,
    get_connection_offers,
    offer_id_for,
)
from backend.copilot.transports import ChatTransportResponse
from backend.data.llm_registry import registry
from backend.util.settings import BehaveAs

USER_ID = "3e53486c-cf57-477e-ba2a-cb02dc828e1a"


def _transport(
    auth_provider: str = "platform",
    credential_id: str | None = None,
    label: str = "AutoGPT Platform",
    available: bool = True,
    default: bool = False,
) -> ChatTransportResponse:
    return ChatTransportResponse(
        auth_provider=auth_provider,  # type: ignore[arg-type]
        credential_id=credential_id,
        label=label,
        available=available,
        default=default,
    )


@pytest.fixture(autouse=True)
def advanced_allowed(mocker: pytest_mock.MockerFixture):
    """Advanced is open unless a test says otherwise."""
    return mocker.patch.object(
        offers, "has_entitlement", new=AsyncMock(return_value=True)
    )


@pytest.fixture(autouse=True)
def hosted(mocker: pytest_mock.MockerFixture):
    mocker.patch.object(offers.settings.config, "behave_as", BehaveAs.CLOUD)
    mocker.patch.object(transports.settings.config, "behave_as", BehaveAs.CLOUD)
    mocker.patch.object(
        provider_tiers, "resolve_use_sdk", new=AsyncMock(return_value=False)
    )
    mocker.patch.object(
        provider_tiers,
        "resolve_model_route",
        new=AsyncMock(
            side_effect=lambda mode, tier, user_id, *, config: SimpleNamespace(
                model=f"{mode}-{tier}-model", source="config"
            )
        ),
    )


def _mock_transports(
    mocker: pytest_mock.MockerFixture, transport_list: list[ChatTransportResponse]
) -> None:
    mocker.patch.object(
        offers,
        "get_chat_transports",
        new=AsyncMock(return_value=transport_list),
    )


@pytest.mark.asyncio
async def test_an_offer_says_what_backs_a_run(
    mocker: pytest_mock.MockerFixture,
) -> None:
    _mock_transports(
        mocker,
        [_transport(default=True), _transport("codex", "cred-1", "ChatGPT")],
    )

    platform, chatgpt = await get_connection_offers(USER_ID)

    assert platform.backed_by_label == "Your AutoGPT plan"
    assert "spend AutoGPT credits" in platform.description
    assert chatgpt.backed_by_label == "Your ChatGPT plan"
    assert "spend no AutoGPT credits" in chatgpt.description


@pytest.mark.asyncio
async def test_self_host_does_not_deny_credits_it_never_had(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch.object(offers.settings.config, "behave_as", BehaveAs.LOCAL)
    _mock_transports(mocker, [_transport(label="Self-hosted chat", default=True)])

    (offer,) = await get_connection_offers(USER_ID)

    assert offer.backed_by_label == "This server's chat provider"
    assert "credits" not in offer.description


@pytest.mark.asyncio
async def test_provider_family_groups_without_becoming_the_credential(
    mocker: pytest_mock.MockerFixture,
) -> None:
    _mock_transports(mocker, [_transport(), _transport("codex", "cred-1", "ChatGPT")])

    platform, chatgpt = await get_connection_offers(USER_ID)

    assert platform.provider_family == "autogpt"
    assert chatgpt.provider_family == "openai"
    assert chatgpt.auth_method == "chatgpt_oauth"
    assert chatgpt.credential_id == "cred-1"


@pytest.mark.asyncio
async def test_offer_ids_are_stable_and_per_account(
    mocker: pytest_mock.MockerFixture,
) -> None:
    first = _transport("codex", "cred-1", "ChatGPT")
    second = _transport("codex", "cred-2", "ChatGPT")
    _mock_transports(mocker, [_transport(), first, second])

    ids = [offer.offer_id for offer in await get_connection_offers(USER_ID)]

    assert ids == ["platform:deployment", "codex:cred-1", "codex:cred-2"]
    assert offer_id_for(first) == offer_id_for(first)


@pytest.mark.asyncio
async def test_a_chatgpt_tier_names_the_model_the_catalog_pins(
    mocker: pytest_mock.MockerFixture,
) -> None:
    # A registry read, not a call to the account: the router validates the
    # pinned slug against what the account advertises when the turn runs.
    mocker.patch.object(
        provider_tiers.llm_registry,
        "get_route",
        side_effect=lambda surface, mode, tier: f"{surface}:{mode}:{tier}",
    )
    _mock_transports(
        mocker, [_transport("platform", None), _transport("codex", "cred-1")]
    )

    codex = [
        o
        for o in await get_connection_offers("user")
        if o.auth_method == "chatgpt_oauth"
    ][0]

    assert [t.display_model for t in codex.tiers] == [
        "copilot_codex:fast:standard",
        "copilot_codex:fast:advanced",
    ]


@pytest.mark.asyncio
async def test_a_chatgpt_tier_the_catalog_pins_nothing_for_stays_unnamed(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch.object(provider_tiers.llm_registry, "get_route", return_value=None)
    _mock_transports(
        mocker, [_transport("platform", None), _transport("codex", "cred-1")]
    )

    codex = [
        o
        for o in await get_connection_offers("user")
        if o.auth_method == "chatgpt_oauth"
    ][0]

    assert all(t.display_model is None for t in codex.tiers)


@pytest.mark.asyncio
async def test_tiers_name_the_model_they_resolve_to(
    mocker: pytest_mock.MockerFixture,
) -> None:
    _mock_transports(mocker, [_transport(default=True)])

    (offer,) = await get_connection_offers(USER_ID)

    assert [tier.label for tier in offer.tiers] == ["Balanced", "Advanced"]
    assert [tier.display_model for tier in offer.tiers] == [
        "fast-standard-model",
        "fast-advanced-model",
    ]


@pytest.mark.asyncio
async def test_tiers_follow_the_engine_the_user_will_actually_run_on(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """The engine is the server's decision, so it is knowable before a turn."""
    mocker.patch.object(
        provider_tiers, "resolve_use_sdk", new=AsyncMock(return_value=True)
    )
    _mock_transports(mocker, [_transport(default=True)])

    (offer,) = await get_connection_offers(USER_ID)

    assert [tier.display_model for tier in offer.tiers] == [
        "thinking-standard-model",
        "thinking-advanced-model",
    ]


@pytest.mark.asyncio
async def test_chatgpt_tiers_are_not_named(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Naming them means leasing a runtime per credential to render a list."""
    _mock_transports(mocker, [_transport("codex", "cred-1", "ChatGPT")])

    (offer,) = await get_connection_offers(USER_ID)

    assert all(tier.display_model is None for tier in offer.tiers)
    assert [tier.label for tier in offer.tiers] == ["Balanced", "Advanced"]


@pytest.mark.asyncio
async def test_an_unresolvable_tier_is_described_without_a_name(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch.object(
        provider_tiers,
        "resolve_model_route",
        new=AsyncMock(side_effect=RuntimeError("registry down")),
    )
    _mock_transports(mocker, [_transport(default=True)])

    (offer,) = await get_connection_offers(USER_ID)

    assert all(tier.display_model is None for tier in offer.tiers)
    assert offer.state == "ready"


@pytest.mark.asyncio
async def test_an_unavailable_connection_is_not_selectable(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch.object(offers.settings.config, "behave_as", BehaveAs.LOCAL)
    chat_config = MagicMock()
    chat_config.test_mode = False
    chat_config.use_claude_code_subscription = False
    chat_config.main_client_credentials = (None, "https://api.anthropic.com/v1/")
    mocker.patch.object(transports, "config", chat_config)
    _mock_transports(mocker, [_transport(label="Self-hosted chat", available=False)])

    (offer,) = await get_connection_offers(USER_ID)

    assert offer.state == "unavailable"
    assert offer.selectable is False
    assert all(tier.selectable is False for tier in offer.tiers)
    assert "No chat provider is configured on this server yet." in offer.limitations


@pytest.mark.asyncio
async def test_chatgpt_states_the_edge_a_user_can_actually_hit(
    mocker: pytest_mock.MockerFixture,
) -> None:
    _mock_transports(mocker, [_transport("codex", "cred-1", "ChatGPT")])

    (offer,) = await get_connection_offers(USER_ID)

    assert offer.limitations == [
        "The agent builder's chat panel always runs on AutoGPT."
    ]


@pytest.mark.asyncio
async def test_exactly_one_offer_is_the_default(
    mocker: pytest_mock.MockerFixture,
) -> None:
    _mock_transports(
        mocker,
        [_transport(default=True), _transport("codex", "cred-1", "ChatGPT")],
    )

    offer_list = await get_connection_offers(USER_ID)

    assert [offer.is_default for offer in offer_list].count(True) == 1


def _upsell(
    mocker: pytest_mock.MockerFixture,
    *,
    entitled: bool = False,
    flag: bool = True,
) -> None:
    mocker.patch.object(
        offers,
        "has_codex_access",
        new=AsyncMock(return_value=entitled),
    )
    mocker.patch.object(offers, "is_feature_enabled", new=AsyncMock(return_value=flag))


def _locked(offer_list: list) -> list:
    return [offer for offer in offer_list if offer.state == "locked"]


@pytest.mark.asyncio
async def test_a_plan_that_excludes_chatgpt_says_so_instead_of_hiding_it(
    mocker: pytest_mock.MockerFixture,
) -> None:
    _mock_transports(mocker, [_transport("platform", None)])
    _upsell(mocker)

    locked = _locked(await get_connection_offers("user"))

    assert len(locked) == 1
    assert locked[0].display_name == "ChatGPT"
    assert locked[0].lock_reason
    assert locked[0].unlock_href == "/settings/billing"


@pytest.mark.asyncio
async def test_a_locked_offer_can_never_be_routed_to(
    mocker: pytest_mock.MockerFixture,
) -> None:
    # The whole risk of showing an unusable connection is that something
    # picks it. It carries no credential and refuses selection, so the
    # picker cannot select it and a turn cannot be addressed to it.
    _mock_transports(mocker, [_transport("platform", None)])
    _upsell(mocker)

    locked = _locked(await get_connection_offers("user"))[0]

    assert locked.selectable is False
    assert locked.credential_id is None
    assert locked.is_default is False


@pytest.mark.asyncio
async def test_a_connected_account_is_never_shadowed_by_an_upsell(
    mocker: pytest_mock.MockerFixture,
) -> None:
    _mock_transports(
        mocker, [_transport("platform", None), _transport("codex", "cred-1")]
    )
    _upsell(mocker, entitled=True)

    assert _locked(await get_connection_offers("user")) == []


@pytest.mark.asyncio
async def test_being_entitled_but_unconnected_is_not_an_upsell(
    mocker: pytest_mock.MockerFixture,
) -> None:
    # Nothing to sell: the plan already includes it, so the invitation to
    # connect belongs on the settings page, not in the composer.
    _mock_transports(mocker, [_transport("platform", None)])
    _upsell(mocker, entitled=True)

    assert _locked(await get_connection_offers("user")) == []


@pytest.mark.asyncio
async def test_entitlement_outage_does_not_show_a_false_upsell(
    mocker: pytest_mock.MockerFixture,
) -> None:
    _mock_transports(mocker, [_transport("platform", None)])
    mocker.patch.object(
        offers,
        "has_codex_access",
        new=AsyncMock(side_effect=RuntimeError("entitlement unavailable")),
    )

    assert _locked(await get_connection_offers("user")) == []


@pytest.mark.asyncio
async def test_the_upsell_stays_off_until_its_cohort_is_opened(
    mocker: pytest_mock.MockerFixture,
) -> None:
    _mock_transports(mocker, [_transport("platform", None)])
    _upsell(mocker, flag=False)

    assert _locked(await get_connection_offers("user")) == []


@pytest.mark.asyncio
async def test_self_host_sells_nothing_because_it_grants_everything(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch.object(offers.settings.config, "behave_as", BehaveAs.LOCAL)
    _mock_transports(mocker, [_transport("platform", None)])
    _upsell(mocker)

    assert _locked(await get_connection_offers("user")) == []


@pytest.mark.asyncio
async def test_advanced_is_locked_for_a_plan_that_does_not_include_it(
    mocker: pytest_mock.MockerFixture, advanced_allowed
) -> None:
    # Visible and locked, not hidden: the row is the upgrade reason.
    advanced_allowed.return_value = False
    _mock_transports(mocker, [_transport("platform", None)])

    tiers = {t.tier: t for t in (await get_connection_offers("user"))[0].tiers}

    assert tiers["standard"].selectable is True
    assert tiers["advanced"].selectable is False
    assert tiers["advanced"].lock_reason
    # Still named, so the user can see what they would be getting.
    assert "advanced" in tiers


@pytest.mark.asyncio
async def test_advanced_is_open_when_the_plan_includes_it(
    mocker: pytest_mock.MockerFixture,
) -> None:
    _mock_transports(mocker, [_transport("platform", None)])

    tiers = {t.tier: t for t in (await get_connection_offers("user"))[0].tiers}

    assert all(t.selectable for t in tiers.values())
    assert all(t.lock_reason is None for t in tiers.values())


@pytest.mark.asyncio
async def test_an_unresolvable_entitlement_leaves_advanced_open(
    mocker: pytest_mock.MockerFixture, advanced_allowed
) -> None:
    # A billing hiccup must not silently downgrade someone's model.
    advanced_allowed.side_effect = RuntimeError("billing down")
    _mock_transports(mocker, [_transport("platform", None)])

    tiers = {t.tier: t for t in (await get_connection_offers("user"))[0].tiers}

    assert tiers["advanced"].selectable is True
    assert tiers["advanced"].lock_reason is None


@pytest.mark.asyncio
async def test_a_locked_chatgpt_offer_still_names_the_models(
    mocker: pytest_mock.MockerFixture,
) -> None:
    # "What you get" is the argument for connecting; a locked row that
    # cannot name the models asks the user to take it on faith.
    mocker.patch.object(
        provider_tiers.llm_registry, "get_route", side_effect=lambda s, m, t: f"{m}-{t}"
    )
    mocker.patch.object(provider_tiers.llm_registry, "get_model", return_value=None)
    _mock_transports(mocker, [_transport("platform", None)])
    _upsell(mocker)

    locked = _locked(await get_connection_offers("user"))[0]

    # The fixture pins the engine to the baseline, so the cells are "fast".
    assert [t.display_model for t in locked.tiers] == [
        "fast-standard",
        "fast-advanced",
    ]
    # Naming them must not read as offering them.
    assert all(t.selectable is False for t in locked.tiers)
    assert locked.selectable is False


@pytest.fixture
def real_catalog():
    """The other tests here run against an empty registry and assert the
    slug fallback. These two are about resolution, so they need the catalog
    actually loaded -- and must put it back, since it is process state."""
    old = (
        registry._dynamic_models,
        registry._date_stripped_models,
        registry._routes,
        registry._loaded,
    )
    registry.load_catalog()
    yield
    (
        registry._dynamic_models,
        registry._date_stripped_models,
        registry._routes,
        registry._loaded,
    ) = old


@pytest.mark.asyncio
async def test_a_platform_tier_names_a_model_configured_in_transport_spelling(
    real_catalog,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """The default configuration routes through OpenRouter, whose slugs carry a
    provider prefix the catalog does not use. Naming the model has to survive
    that, or the row reads as a routing key on every deployment that took the
    default."""
    _mock_transports(mocker, [_transport(default=True)])
    mocker.patch.object(
        provider_tiers,
        "resolve_model_route",
        new=AsyncMock(
            side_effect=lambda mode, tier, user_id, *, config: SimpleNamespace(
                model="anthropic/claude-sonnet-5", source="config"
            )
        ),
    )

    offer = (await get_connection_offers(USER_ID))[0]

    assert [tier.display_model for tier in offer.tiers] == [
        "Claude Sonnet 5",
        "Claude Sonnet 5",
    ]


@pytest.mark.asyncio
async def test_a_model_the_catalog_does_not_know_is_named_by_its_slug(
    real_catalog,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """A deployment may point a tier at anything. Showing the slug is worse than
    showing a name and better than showing nothing -- without the vendor
    prefix, which is addressing rather than identity."""
    _mock_transports(mocker, [_transport(default=True)])
    mocker.patch.object(
        provider_tiers,
        "resolve_model_route",
        new=AsyncMock(
            side_effect=lambda mode, tier, user_id, *, config: SimpleNamespace(
                model="acme/a-model-nobody-has-heard-of", source="config"
            )
        ),
    )

    offer = (await get_connection_offers(USER_ID))[0]

    assert offer.tiers[0].display_model == "a-model-nobody-has-heard-of"


class TestAdvancedTierEntitlement:
    """The picker and the turn ask the same question and need opposite answers
    when the entitlement service is down: show the tier, refuse the spend."""

    async def test_entitled_reports_what_the_service_said(
        self, mocker: pytest_mock.MockFixture
    ) -> None:
        mocker.patch.object(offers, "has_entitlement", new=AsyncMock(return_value=True))
        assert await advanced_tier_entitled(USER_ID) is True

        mocker.patch.object(
            offers, "has_entitlement", new=AsyncMock(return_value=False)
        )
        assert await advanced_tier_entitled(USER_ID) is False

    async def test_enforcement_refuses_to_guess_when_lookup_fails(
        self, mocker: pytest_mock.MockFixture
    ) -> None:
        # Returning True here is what let an unentitled account spend on
        # Advanced during a billing outage. Not knowing is not permission.
        mocker.patch.object(
            offers,
            "has_entitlement",
            new=AsyncMock(side_effect=RuntimeError("billing down")),
        )
        with pytest.raises(EntitlementUnavailable):
            await advanced_tier_entitled(USER_ID)

    async def test_the_picker_stays_generous_when_lookup_fails(
        self, mocker: pytest_mock.MockFixture
    ) -> None:
        # The control staying on offer costs nothing, because the turn is
        # enforced separately.
        mocker.patch.object(
            offers,
            "has_entitlement",
            new=AsyncMock(side_effect=RuntimeError("billing down")),
        )
        assert await advanced_tier_allowed(USER_ID) is True
