"""Which named services the platform can actually connect."""

import pytest
from pytest_mock import MockerFixture

from backend.api.features.integrations import support
from backend.api.features.integrations.models import get_supported_auth_types
from backend.api.features.integrations.support import ConnectableService
from backend.integrations.mcp_catalog import get_mcp_catalog
from backend.integrations.service_identity import catalog_slug

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


def test_live_registry_offers_printify_and_every_mcp_catalog_server():
    keys = {service.key for service in support.connectable_services()}
    assert "printify" in keys
    assert {catalog_slug(entry) for entry in get_mcp_catalog()} <= keys
    assert "mcp" not in keys  # plumbing, not a service anyone names


def test_a_provider_counts_exactly_when_it_has_a_sign_in_here():
    """Etsy's blocks always exist; whether it is supported follows whether
    this deployment configured its OAuth app, whichever way the env is set."""
    has_sign_in = bool(get_supported_auth_types("etsy"))
    assert (support.find_connectable("Etsy") is not None) == has_sign_in


def test_the_registry_is_read_once_per_batch(mocker: MockerFixture):
    """Labelling a page of tasks must not rebuild the registry per name."""
    read = mocker.patch.object(
        support, "connectable_services", return_value=FAKE_REGISTRY
    )

    support.unsupported_services(["Shopify", "Printify", "Etsy", "Google Sheets"])

    assert read.call_count == 1


def test_a_known_tool_is_named_one_way_however_it_was_written(registry):
    assert support.canonical_name("Shopify store") == "Shopify"
    assert support.canonical_name("shopify") == "Shopify"
    assert support.canonical_name("Gelato API") == "Gelato"
    assert support.canonical_name("  Big  Cartel ") == "Big Cartel"


def test_a_word_shared_with_a_domain_name_is_not_that_service(registry):
    assert support.find_connectable("Customer support inbox") is None
