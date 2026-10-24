"""Which named services the platform can actually connect."""

import pytest
from pytest_mock import MockerFixture

from backend.api.features.integrations import support
from backend.api.features.integrations.support import ConnectableService

FAKE_REGISTRY = [
    ConnectableService(key="google", name="Google"),
    ConnectableService(key="printify", name="Printify"),
    ConnectableService(key="canva", name="Canva"),
    ConnectableService(key="monday", name="monday.com"),
    ConnectableService(key="customer_io", name="Customer.io (US)"),
]


@pytest.fixture
def registry(mocker: MockerFixture) -> None:
    mocker.patch.object(support, "connectable_services", return_value=FAKE_REGISTRY)


@pytest.mark.parametrize(
    "name, key",
    [
        ("printify", "printify"),
        ("Printify", "printify"),
        ("Printify API", "printify"),
        ("Google Drive", "google"),
        ("google_sheets", "google"),
        ("Monday", "monday"),
        ("customer.io", "customer_io"),
    ],
)
def test_a_service_is_found_by_id_name_or_product(registry, name: str, key: str):
    found = support.find_connectable(name)
    assert found is not None and found.key == key


def test_unsupported_keeps_the_names_we_cannot_connect_in_order(registry):
    assert support.unsupported_services(
        ["printify", "Shopify", "Etsy", "shopify", "Google Sheets", " "]
    ) == ["Shopify", "Etsy"]


def test_mentioned_gaps_finds_known_unsupported_tools_in_free_text(registry):
    text = "List each sticker on Shopify and Printful, then post it to Canva."
    assert support.mentioned_gaps(text) == ["Shopify", "Printful"]


def test_mentioned_gaps_ignores_words_that_only_contain_a_name(registry):
    assert support.mentioned_gaps("A wixed-up shopifyish idea for gelatos") == []


def test_live_registry_offers_printify_and_the_mcp_catalog():
    keys = {service.key for service in support.connectable_services()}
    assert "printify" in keys
    assert "canva" in keys  # an official MCP catalog server
    assert "mcp" not in keys  # plumbing, not a service anyone names


def test_a_provider_without_a_sign_in_here_is_not_connectable():
    # No ETSY_CLIENT_ID/SECRET in the test environment: Etsy has blocks but
    # no way to connect, so it must not count as supported.
    assert support.find_connectable("Etsy") is None


def test_a_word_shared_with_a_domain_name_is_not_that_service(registry):
    assert support.find_connectable("Customer support inbox") is None
