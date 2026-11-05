"""Printify client and blocks against a mocked HTTP layer (no real API calls)."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import ValidationError

from backend.blocks.printify._api import PrintifyClient, PrintifyError
from backend.blocks.printify._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.printify.catalog import PrintifySearchCatalogBlock
from backend.blocks.printify.products import PrintifyCreateProductBlock, product_body
from backend.blocks.printify.uploads import PrintifyUploadImageBlock
from backend.copilot.tools.utils import credential_rejection_status
from backend.data.execution import ExecutionContext
from backend.util import json
from backend.util.request import HTTPClientError, HTTPServerError


def _response(status: int, body: object) -> Mock:
    content = json.dumps(body).encode()
    response = Mock(ok=200 <= status < 300, status=status, content=content)
    response.json.return_value = body
    return response


def _client(*responses: Mock) -> tuple[PrintifyClient, AsyncMock]:
    client = PrintifyClient(TEST_CREDENTIALS)
    request = AsyncMock(side_effect=list(responses))
    client._requests.request = request
    client._once.request = request
    return client, request


async def test_requests_carry_the_token_and_a_user_agent():
    client, request = _client(
        _response(200, [{"id": 1, "title": "Stickers", "sales_channel": "etsy"}])
    )

    shops = await client.list_shops()

    assert [shop.title for shop in shops] == ["Stickers"]
    call = request.await_args
    assert call.args == ("GET", "https://api.printify.com/v1/shops.json")
    assert call.kwargs["headers"]["Authorization"] == "Bearer mock-printify-token"
    assert call.kwargs["headers"]["User-Agent"] == "AutoGPT"


async def test_upload_sends_a_url_or_base64_contents():
    image = {"id": "img-1", "file_name": "cat.png", "preview_url": "https://p/1"}
    client, request = _client(_response(200, image), _response(200, image))

    await client.upload_image("cat.png", url="https://example.com/cat.png")
    await client.upload_image("cat.png", contents="aGVsbG8=")

    by_url, by_contents = (call.kwargs["json"] for call in request.await_args_list)
    assert by_url == {"file_name": "cat.png", "url": "https://example.com/cat.png"}
    assert by_contents == {"file_name": "cat.png", "contents": "aGVsbG8="}


async def test_a_rejected_token_says_where_to_make_a_new_one():
    client, _ = _client(_response(401, {"message": "Unauthenticated."}))

    with pytest.raises(PrintifyError, match="printify.com/app/account/api"):
        await client.list_shops()


async def test_a_validation_error_keeps_printifys_reason():
    client, _ = _client(
        _response(
            400,
            {
                "status": "error",
                "message": "Validation failed.",
                "errors": {"reason": "Image has low quality", "code": 8203},
            },
        )
    )

    with pytest.raises(PrintifyError, match="Validation failed. Image has low quality"):
        await client.create_product(1, {})


def _product_input(price: float) -> PrintifyCreateProductBlock.Input:
    return PrintifyCreateProductBlock.Input(
        credentials=TEST_CREDENTIALS_INPUT,
        shop_id=1,
        title="Sleepy Cat",
        description="A cat.",
        tags=["cat"],
        blueprint_id=400,
        print_provider_id=99,
        variant_ids=[11],
        price=price,
        image_id="img-1",
    )


@pytest.mark.parametrize("price", [0, 0.004, -1.0])
def test_a_price_that_rounds_under_a_cent_is_refused(price: float):
    """0.004 rounds to 0 cents: Printify would list the product for free."""
    with pytest.raises(ValueError, match="at least 0.01"):
        product_body(_product_input(price))


@pytest.mark.parametrize("price, cents", [(0.005, 1), (1.005, 101), (3.99, 399)])
def test_a_price_is_sent_in_whole_cents_rounded_half_up(price: float, cents: int):
    [variant] = product_body(_product_input(price))["variants"]
    assert variant["price"] == cents


def test_product_body_prices_every_variant_in_cents_and_centres_the_design():
    data = PrintifyCreateProductBlock.Input(
        credentials=TEST_CREDENTIALS_INPUT,
        shop_id=1,
        title="Sleepy Cat",
        description="A cat.",
        tags=["cat"],
        blueprint_id=400,
        print_provider_id=99,
        variant_ids=[11, 12],
        price=3.99,
        image_id="img-1",
    )

    body = product_body(data)

    assert body["variants"] == [
        {"id": 11, "price": 399, "is_enabled": True},
        {"id": 12, "price": 399, "is_enabled": True},
    ]
    assert body["print_areas"] == [
        {
            "variant_ids": [11, 12],
            "placeholders": [
                {
                    "position": "front",
                    "images": [
                        {"id": "img-1", "x": 0.5, "y": 0.5, "scale": 1, "angle": 0}
                    ],
                }
            ],
        }
    ]


async def test_search_catalog_matches_singular_and_plural_names():
    block = PrintifySearchCatalogBlock()
    blueprints = [
        {"id": 400, "title": "Kiss-Cut Stickers"},
        {"id": 5, "title": "Unisex Heavy Cotton Tee"},
    ]
    for query in ("stickers", "sticker", "Kiss-Cut sticker"):
        client, _ = _client(_response(200, blueprints))
        with patch(
            "backend.blocks.printify.catalog.PrintifyClient", return_value=client
        ):
            outputs = [
                output
                async for output in block.run(
                    block.Input(credentials=TEST_CREDENTIALS_INPUT, query=query),
                    credentials=TEST_CREDENTIALS,
                )
            ]

        assert [b.id for b in outputs[0][1]] == [400], query


@pytest.mark.parametrize(
    "query, expected",
    [
        ("glass", [7]),  # not "gla": a double s is not a plural
        ("boss", []),  # not "bo", which is in "Bottle"
        ("ads", []),  # not "ad", which is in "Mouse Pad"
        ("stickers", [400]),
    ],
)
async def test_search_catalog_takes_off_one_plural_s_at_most(
    query: str, expected: list[int]
):
    block = PrintifySearchCatalogBlock()
    blueprints = [
        {"id": 400, "title": "Kiss-Cut Stickers"},
        {"id": 7, "title": "Glass Mug"},
        {"id": 9, "title": "Water Bottle"},
        {"id": 13, "title": "Mouse Pad"},
    ]
    client, _ = _client(_response(200, blueprints))
    with patch("backend.blocks.printify.catalog.PrintifyClient", return_value=client):
        outputs = [
            output
            async for output in block.run(
                block.Input(credentials=TEST_CREDENTIALS_INPUT, query=query),
                credentials=TEST_CREDENTIALS,
            )
        ]

    assert [b.id for b in outputs[0][1]] == expected


async def test_upload_block_sends_a_file_as_base64_and_a_url_as_is():
    block = PrintifyUploadImageBlock()
    upload = AsyncMock(return_value=Mock(id="img", preview_url="", width=1, height=1))
    context = ExecutionContext(user_id="user-1", graph_exec_id="run-1")
    with (
        patch.object(PrintifyUploadImageBlock, "upload", upload),
        patch(
            "backend.blocks.printify.uploads.store_media_file",
            AsyncMock(return_value="data:image/png;base64,aGVsbG8="),
        ),
    ):
        for image in ("workspace://file-1", "https://example.com/d/cat.png"):
            async for _ in block.run(
                block.Input(credentials=TEST_CREDENTIALS_INPUT, image=image),
                credentials=TEST_CREDENTIALS,
                execution_context=context,
            ):
                pass

    from_file, from_url = (call.args for call in upload.await_args_list)
    assert from_file == (TEST_CREDENTIALS, "design.png", None, "aGVsbG8=")
    assert from_url == (
        TEST_CREDENTIALS,
        "cat.png",
        "https://example.com/d/cat.png",
        None,
    )


async def test_a_rejected_token_reaches_the_reconnect_path():
    """The shared recovery reads the HTTP status off the exception; a
    friendly message must not cost the user the reconnect card."""
    client, _ = _client(_response(401, {"message": "Unauthenticated."}))

    with pytest.raises(HTTPClientError) as caught:
        await client.list_shops()

    assert caught.value.status_code == 401
    assert credential_rejection_status(caught.value) == 401


async def test_a_printify_outage_is_a_server_error_not_a_client_one():
    client, _ = _client(_response(503, {"message": "Down"}))

    with pytest.raises(HTTPServerError) as caught:
        await client.list_shops()

    assert caught.value.status_code == 503


def test_retries_on_throttling_are_bounded():
    assert PrintifyClient(TEST_CREDENTIALS)._requests.retry_max_attempts == 3


async def test_only_reads_are_retried():
    """A 5xx after Printify already created the product must not create a
    second one: anything that writes goes out once."""
    client = PrintifyClient(TEST_CREDENTIALS)
    assert client._requests.retry_max_attempts == 3
    assert client._once.retry_max_attempts == 1
    image = {"id": "img-1"}
    product = {"id": "prod-1", "title": "Cat"}
    client._requests.request = AsyncMock(return_value=_response(200, []))
    client._once.request = AsyncMock(
        side_effect=[_response(200, image), _response(200, product), _response(200, {})]
    )

    await client.list_shops()
    await client.upload_image("cat.png", url="https://example.com/cat.png")
    await client.create_product(1, {})
    await client.publish_product(1, "prod-1")

    assert [c.args[0] for c in client._requests.request.await_args_list] == ["GET"]
    assert [c.args[0] for c in client._once.request.await_args_list] == [
        "POST",
        "POST",
        "POST",
    ]


@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.list_variants(400, 99),
        lambda c: c.list_products(1, 20),
    ],
    ids=["variants", "products"],
)
async def test_a_malformed_success_body_is_a_validation_error(call):
    client, _ = _client(_response(200, {"unexpected": True}))

    with pytest.raises(ValidationError):
        await call(client)
