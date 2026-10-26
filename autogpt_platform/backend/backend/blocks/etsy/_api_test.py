"""Etsy client, OAuth handler and listing forms against a mocked HTTP layer."""

import time
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import parse_qs, urlsplit

import pytest
from pydantic import ValidationError

from backend.blocks.etsy._api import EtsyClient, EtsyError
from backend.blocks.etsy._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.etsy._oauth import EtsyOAuthHandler
from backend.blocks.etsy.images import decode_data_uri
from backend.blocks.etsy.listings import (
    EtsyCreateDraftListingBlock,
    EtsyUpdateListingBlock,
    draft_listing_form,
    update_listing_form,
)
from backend.copilot.tools.utils import credential_rejection_status
from backend.util import json
from backend.util.request import HTTPClientError, HTTPServerError

SHOP = {"shop_id": 42, "shop_name": "StickerStudio", "url": "https://etsy.com/shop/s"}


def _response(status: int, body: object) -> Mock:
    response = Mock(ok=200 <= status < 300, status=status)
    response.content = json.dumps(body).encode()
    response.json.return_value = body
    return response


def _http(*responses: Mock) -> Mock:
    http = Mock()
    http.request = AsyncMock(side_effect=list(responses))
    http.post = AsyncMock(side_effect=list(responses))
    return http


def _client(*responses: Mock) -> tuple[EtsyClient, Mock]:
    http = _http(*responses)
    with patch.object(EtsyClient, "raw_requests", return_value=http):
        return EtsyClient(TEST_CREDENTIALS, "keystring", "shared-secret"), http


async def test_every_request_sends_the_app_key_pair_and_the_shop_token():
    client, http = _client(
        _response(200, {"user_id": 7, "shop_id": 42}), _response(200, SHOP)
    )

    shop = await client.get_my_shop()

    assert shop.shop_name == "StickerStudio"
    me, shop_call = http.request.await_args_list
    assert me.args == ("GET", "https://api.etsy.com/v3/application/users/me")
    assert shop_call.args == ("GET", "https://api.etsy.com/v3/application/shops/42")
    headers = me.kwargs["headers"]
    assert headers["x-api-key"] == "keystring:shared-secret"
    assert headers["Authorization"] == "Bearer 12345678.mock-etsy-access-token"


async def test_an_account_without_a_shop_is_told_to_open_one():
    client, _ = _client(_response(200, {"user_id": 7, "shop_id": None}))

    with pytest.raises(EtsyError, match="no shop yet"):
        await client.get_my_shop()


async def test_a_signed_out_connection_asks_to_reconnect():
    client, _ = _client(_response(401, {"error": "invalid_token"}))

    with pytest.raises(EtsyError, match="Connect Etsy again"):
        await client.list_listings(42, "active", 25)


async def test_categories_are_flattened_with_their_path():
    tree = {
        "count": 1,
        "results": [
            {
                "id": 1,
                "name": "Paper",
                "children": [{"id": 2078, "name": "Stickers", "children": []}],
            }
        ],
    }
    client, _ = _client(_response(200, tree))

    categories = await client.list_categories()

    assert [(c.id, c.path) for c in categories] == [
        (1, "Paper"),
        (2078, "Paper > Stickers"),
    ]


async def test_listing_image_is_sent_as_the_image_multipart_field():
    client, http = _client(_response(200, {"listing_image_id": 9}))

    await client.upload_listing_image(
        42, 5, ("image.png", b"png", "image/png"), {"rank": "1"}
    )

    call = http.request.await_args
    assert call.args[1].endswith("/shops/42/listings/5/images")
    [(field, (name, handle, mime))] = call.kwargs["files"]
    assert (field, name, handle.read(), mime) == (
        "image",
        "image.png",
        b"png",
        "image/png",
    )
    assert call.kwargs["data"] == {"rank": "1"}


def _draft(**overrides) -> EtsyCreateDraftListingBlock.Input:
    fields = {
        "credentials": TEST_CREDENTIALS_INPUT,
        "shop_id": 42,
        "title": "Sleepy Cat Sticker",
        "description": "A cat.",
        "price": 3.99,
        "tags": ["cat sticker", "kawaii"],
        "taxonomy_id": 2078,
        "shipping_profile_id": 2001,
        **overrides,
    }
    return EtsyCreateDraftListingBlock.Input(**fields)


def test_draft_listing_form_matches_what_etsy_reads():
    assert draft_listing_form(_draft()) == {
        "title": "Sleepy Cat Sticker",
        "description": "A cat.",
        "price": "3.99",
        "quantity": "999",
        "taxonomy_id": "2078",
        "who_made": "i_did",
        "when_made": "made_to_order",
        "is_supply": "false",
        "type": "physical",
        "tags": "cat sticker,kawaii",
        "shipping_profile_id": "2001",
    }


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"tags": [f"t{i}" for i in range(14)]}, "at most 13 tags"),
        ({"price": 0}, "at least 0.01"),
        ({"price": 0.004}, "at least 0.01"),  # "0.00" on the form
    ],
)
def test_draft_listing_form_refuses_what_etsy_would(overrides, message):
    with pytest.raises(ValueError, match=message):
        draft_listing_form(_draft(**overrides))


@pytest.mark.parametrize(
    "price, sent", [(0.005, "0.01"), (1.005, "1.01"), (3.5, "3.50")]
)
def test_a_draft_price_is_sent_in_whole_cents(price: float, sent: str):
    assert draft_listing_form(_draft(price=price))["price"] == sent


@pytest.mark.parametrize("listing_type", ["physical", "download"])
def test_a_draft_needs_no_shipping_profile(listing_type):
    """Etsy dropped the shipping-profile requirement for every draft
    (etsy/open-api discussion 1524, Jan 2026): a physical draft without
    one is sent, with the field left out."""
    form = draft_listing_form(_draft(shipping_profile_id=0, listing_type=listing_type))
    assert "shipping_profile_id" not in form


def test_update_form_sends_only_what_changes():
    data = EtsyUpdateListingBlock.Input(
        credentials=TEST_CREDENTIALS_INPUT, shop_id=42, listing_id=5, state="active"
    )
    assert update_listing_form(data) == {"state": "active"}

    with pytest.raises(ValueError, match="Nothing to change"):
        update_listing_form(
            EtsyUpdateListingBlock.Input(
                credentials=TEST_CREDENTIALS_INPUT, shop_id=42, listing_id=5
            )
        )


def test_decode_data_uri():
    assert decode_data_uri("data:image/jpeg;base64,aGVsbG8=") == (
        "image.jpg",
        b"hello",
        "image/jpeg",
    )
    with pytest.raises(ValueError, match="could not be read"):
        decode_data_uri("https://example.com/cat.png")


def _handler() -> EtsyOAuthHandler:
    return EtsyOAuthHandler(
        "keystring",
        "shared-secret",
        "https://app.example/auth/integrations/oauth_callback",
    )


def test_login_url_uses_pkce_and_space_separated_scopes():
    url = _handler().get_login_url(["shops_r", "listings_w"], "state-1", "challenge")

    parts = urlsplit(url)
    query = parse_qs(parts.query)
    assert (
        f"{parts.scheme}://{parts.netloc}{parts.path}"
        == "https://www.etsy.com/oauth/connect"
    )
    assert query["scope"] == ["shops_r listings_w"]
    assert query["code_challenge"] == ["challenge"]
    assert query["code_challenge_method"] == ["S256"]
    assert query["client_id"] == ["keystring"]

    with pytest.raises(ValueError, match="PKCE"):
        _handler().get_login_url([], "state-1", None)


async def test_code_exchange_stores_tokens_and_names_the_shop():
    tokens = {
        "access_token": "7.access",
        "refresh_token": "7.refresh",
        "expires_in": 3600,
        "token_type": "Bearer",
    }
    http = _http(_response(200, tokens))
    http.request = AsyncMock(
        side_effect=[
            _response(200, {"user_id": 7, "shop_id": 42}),
            _response(200, SHOP),
        ]
    )
    with patch.object(EtsyClient, "raw_requests", return_value=http):
        credentials = await _handler().exchange_code_for_tokens(
            "code-1", ["shops_r"], "verifier-1"
        )

    form = http.post.await_args.kwargs["data"]
    assert form["grant_type"] == "authorization_code"
    assert form["code_verifier"] == "verifier-1"
    assert credentials.access_token.get_secret_value() == "7.access"
    assert credentials.username == "StickerStudio"
    assert credentials.access_token_expires_at is not None
    assert credentials.access_token_expires_at > time.time() + 3000


async def test_refresh_keeps_the_credential_identity():
    tokens = {
        "access_token": "7.new",
        "refresh_token": "7.new-refresh",
        "expires_in": 3600,
    }
    http = _http(_response(200, tokens))
    with patch.object(EtsyClient, "raw_requests", return_value=http):
        refreshed = await _handler().refresh_tokens(TEST_CREDENTIALS)

    assert http.post.await_args.kwargs["data"]["grant_type"] == "refresh_token"
    assert refreshed.id == TEST_CREDENTIALS.id
    assert refreshed.username == "StickerStudio"
    assert refreshed.refresh_token is not None
    assert refreshed.refresh_token.get_secret_value() == "7.new-refresh"


async def test_a_signed_out_shop_reaches_the_reconnect_path():
    client, _ = _client(_response(401, {"error": "invalid_token"}))

    with pytest.raises(HTTPClientError) as caught:
        await client.get_my_shop()

    assert credential_rejection_status(caught.value) == 401


async def test_a_refused_refresh_is_a_client_error_the_card_handles():
    """creds_manager hands an HTTPClientError from a refresh to the
    reconnect card; anything else becomes a bare failure."""
    http = _http(_response(400, {"error": "invalid_grant"}))
    with (
        patch.object(EtsyClient, "raw_requests", return_value=http),
        pytest.raises(HTTPClientError) as caught,
    ):
        await _handler().refresh_tokens(TEST_CREDENTIALS)

    assert caught.value.status_code == 400


async def test_an_etsy_outage_is_a_server_error():
    client, _ = _client(_response(502, {"error": "bad gateway"}))

    with pytest.raises(HTTPServerError):
        await client.list_listings(42, "active", 25)


def test_retries_on_throttling_are_bounded():
    assert EtsyClient.raw_requests().retry_max_attempts == 3


async def test_only_reads_and_edits_are_retried():
    """A 5xx after Etsy already made the draft (or took the photo) must not
    make a second one; the token exchange's code is single-use too."""
    retried, once = _http(_response(200, {"count": 0, "results": []})), _http(
        _response(200, {"listing_id": 5, "title": "Cat"})
    )
    with patch.object(
        EtsyClient,
        "raw_requests",
        side_effect=lambda attempts=3: retried if attempts == 3 else once,
    ) as factory:
        client = EtsyClient(TEST_CREDENTIALS, "keystring", "shared-secret")
        await client.list_listings(42, "active", 25)
        await client.create_draft_listing(42, {"title": "Cat"})

    assert {call.kwargs.get("attempts", 3) for call in factory.call_args_list} == {3, 1}
    assert [c.args[0] for c in retried.request.await_args_list] == ["GET"]
    assert [c.args[0] for c in once.request.await_args_list] == ["POST"]
    assert EtsyClient.raw_requests(attempts=1).retry_max_attempts == 1


@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.list_listings(42, "active", 25),
        lambda c: c.list_shipping_profiles(42),
        lambda c: c.list_categories(),
        lambda c: c.list_return_policies(42),
    ],
    ids=["listings", "shipping_profiles", "categories", "return_policies"],
)
async def test_a_malformed_success_body_is_a_validation_error(call):
    """A block reports a ValidationError as an execution error; a KeyError
    would read as an unknown platform fault."""
    client, _ = _client(_response(200, {"unexpected": True}))

    with pytest.raises(ValidationError):
        await call(client)


async def test_processing_profiles_are_listed_for_physical_listings():
    profiles = {
        "count": 1,
        "results": [
            {
                "shop_id": 42,
                "readiness_state_id": 7001,
                "readiness_state": "made_to_order",
                "min_processing_days": 1,
                "max_processing_days": 3,
                "processing_days_display_label": "1-3 days",
            }
        ],
    }
    client, http = _client(_response(200, profiles))

    result = await client.list_processing_profiles(42)

    assert http.request.await_args.args[1].endswith(
        "/shops/42/readiness-state-definitions"
    )
    assert [(p.readiness_state_id, p.label) for p in result] == [(7001, "1-3 days")]


def test_a_processing_profile_is_sent_when_given():
    assert draft_listing_form(_draft(readiness_state_id=7001))[
        "readiness_state_id"
    ] == ("7001")
    assert "readiness_state_id" not in draft_listing_form(_draft())


async def test_a_shop_name_that_cannot_be_read_never_fails_the_sign_in():
    """The code is single-use and already spent: a bad shop reply may cost
    the account its label, never the connection."""
    tokens = {
        "access_token": "7.access",
        "refresh_token": "7.refresh",
        "expires_in": 3600,
    }
    http = _http(_response(200, tokens))
    http.request = AsyncMock(return_value=_response(200, ["not", "an", "object"]))
    with patch.object(EtsyClient, "raw_requests", return_value=http):
        credentials = await _handler().exchange_code_for_tokens(
            "code-1", ["shops_r"], "verifier-1"
        )

    assert credentials.access_token.get_secret_value() == "7.access"
    assert credentials.username is None


# getShopReturnPolicies, as Etsy's Open API v3 documents its 200 body
# (ShopReturnPolicies: count and results of ShopReturnPolicy).
RETURN_POLICIES = {
    "count": 2,
    "results": [
        {
            "return_policy_id": 9001,
            "shop_id": 42,
            "accepts_returns": True,
            "accepts_exchanges": True,
            "return_deadline": 30,
        },
        {
            "return_policy_id": 9002,
            "shop_id": 42,
            "accepts_returns": False,
            "accepts_exchanges": False,
            "return_deadline": None,
        },
    ],
}


async def test_a_shops_return_policies_are_listed():
    client, http = _client(_response(200, RETURN_POLICIES))

    policies = await client.list_return_policies(42)

    assert http.request.await_args.args == (
        "GET",
        "https://api.etsy.com/v3/application/shops/42/policies/return",
    )
    assert [
        (p.return_policy_id, p.accepts_returns, p.return_deadline) for p in policies
    ] == [
        (9001, True, 30),
        (9002, False, None),
    ]


def test_a_return_policy_goes_on_the_draft_when_given():
    assert draft_listing_form(_draft(return_policy_id=9001))["return_policy_id"] == (
        "9001"
    )
    assert "return_policy_id" not in draft_listing_form(_draft())


def test_publishing_can_set_the_shipping_profile_and_return_policy():
    """Etsy requires both on an active physical listing (a return policy
    outside the EU); a draft made without them is completed on publish."""
    data = EtsyUpdateListingBlock.Input(
        credentials=TEST_CREDENTIALS_INPUT,
        shop_id=42,
        listing_id=5,
        state="active",
        shipping_profile_id=2001,
        return_policy_id=9001,
    )

    assert update_listing_form(data) == {
        "state": "active",
        "shipping_profile_id": "2001",
        "return_policy_id": "9001",
    }


async def test_a_refused_publish_says_what_etsy_needs():
    """Etsy answers with a 400 and its own reason; the block keeps it and
    adds what a physical listing needs to go live, and where to get it."""
    refused = _response(
        400, {"error": "A return policy is required for active physical listings."}
    )
    block = EtsyUpdateListingBlock()
    with patch.object(EtsyClient, "raw_requests", return_value=_http(refused)):
        with pytest.raises(EtsyError) as raised:
            async for _ in block.run(
                EtsyUpdateListingBlock.Input(
                    credentials=TEST_CREDENTIALS_INPUT,
                    shop_id=42,
                    listing_id=5,
                    state="active",
                ),
                credentials=TEST_CREDENTIALS,
            ):
                pass

    message = str(raised.value)
    assert "A return policy is required for active physical listings." in message
    assert "return policy" in message and "Get My Etsy Shop" in message
    assert raised.value.status_code == 400


async def test_a_refused_edit_that_is_not_a_publish_is_left_as_etsy_said():
    refused = _response(400, {"error": "Title is too long."})
    block = EtsyUpdateListingBlock()
    with patch.object(EtsyClient, "raw_requests", return_value=_http(refused)):
        with pytest.raises(EtsyError, match="Title is too long.$"):
            async for _ in block.run(
                EtsyUpdateListingBlock.Input(
                    credentials=TEST_CREDENTIALS_INPUT,
                    shop_id=42,
                    listing_id=5,
                    title="x",
                ),
                credentials=TEST_CREDENTIALS,
            ):
                pass


@pytest.mark.parametrize(
    "days, label",
    [
        ((1, 3), "1-3 days"),
        ((2, 2), "2 days"),
        ((1, 1), "1 day"),
        ((5, None), "5 days"),
    ],
)
async def test_a_processing_profile_without_etsys_label_says_its_days(days, label):
    """getShopReadinessStateDefinitions marks no field required, so the
    translated label can be missing; the days it documents still say it."""
    low, high = days
    profile = {"shop_id": 42, "readiness_state_id": 7002, "min_processing_days": low}
    if high is not None:
        profile["max_processing_days"] = high
    client, _ = _client(_response(200, {"count": 1, "results": [profile]}))

    [result] = await client.list_processing_profiles(42)

    assert (result.min_processing_days, result.label) == (low, label)
