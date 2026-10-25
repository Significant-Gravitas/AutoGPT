"""Which named services the platform can actually connect."""

import pytest
from pytest_mock import MockerFixture

from backend.api.features.integrations import block_coverage, support
from backend.api.features.integrations.models import get_supported_auth_types
from backend.api.features.integrations.support import ConnectableService
from backend.data.model import CredentialsFieldInfo
from backend.integrations.mcp_catalog import get_mcp_catalog
from backend.integrations.oauth.app_credentials import oauth_client_credentials
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


def test_a_task_text_naming_known_unsupported_tools_is_read(registry):
    text = "List each sticker on Shopify and Printful, then post it to Canva."
    assert support.unsupported_in_task(None, text) == ["Shopify", "Printful"]


def test_a_task_text_ignores_words_that_only_contain_a_name(registry):
    text = "A wixed-up shopifyish idea for gelatos"
    assert support.unsupported_in_task([], text) == []


def test_live_registry_offers_printify_and_every_mcp_catalog_server():
    keys = {service.key for service in support.connectable_services()}
    assert "printify" in keys
    assert {catalog_slug(entry) for entry in get_mcp_catalog()} <= keys
    assert "mcp" not in keys  # plumbing, not a service anyone names


def test_a_provider_counts_exactly_when_it_has_a_sign_in_here():
    """Etsy's blocks always exist; whether it is supported follows whether
    this deployment configured its OAuth app, whichever way the env is set."""
    found = support.find_connectable("Etsy")  # loads the blocks first
    configured = oauth_client_credentials("etsy") is not None
    assert bool(get_supported_auth_types("etsy")) == configured
    assert (found is not None) == configured


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


def _without_oauth_apps(mocker: MockerFixture) -> None:
    mocker.patch.object(support, "oauth_client_credentials", return_value=None)


def test_an_oauth_only_provider_needs_its_app_configured(mocker: MockerFixture):
    """Google declares OAuth even where no app is set up; it only counts once
    a user here could actually finish the sign-in."""
    _without_oauth_apps(mocker)
    assert support.find_connectable("Google Sheets") is None
    assert support.find_connectable("Reddit") is None

    mocker.patch.object(
        support, "oauth_client_credentials", return_value=("id", "secret")
    )
    found = support.find_connectable("Google Sheets")
    assert found is not None and found.key == "google"


def test_a_provider_users_can_bring_a_key_for_needs_no_app(mocker: MockerFixture):
    """GitHub also takes a personal token, so a missing OAuth app does not
    make it unsupported."""
    _without_oauth_apps(mocker)
    found = support.find_connectable("GitHub")
    assert found is not None and found.key == "github"


def test_a_legacy_connector_without_provider_metadata_is_connectable():
    """Slack declares its bot-token credentials only on its blocks' fields
    (blocks/slack/_auth.py), with no ProviderBuilder: it is still supported."""
    assert get_supported_auth_types("slack") == []

    found = support.find_connectable("Slack")

    assert found is not None and found.key == "slack"
    assert support.unsupported_services(["Slack"]) == []


def test_a_credential_type_counts_only_from_a_block_that_can_run(
    mocker: MockerFixture,
):
    """A disabled block takes no credentials from anyone, so its provider
    gains no sign-in from it."""

    info = CredentialsFieldInfo.model_validate(
        {
            "credentials_provider": ["acme_legacy"],
            "credentials_types": ["api_key"],
        }
    )

    def block(disabled: bool):
        instance = mocker.Mock(disabled=disabled)
        instance.name = "AcmeSendBlock"
        instance.input_schema.get_credentials_fields_info.return_value = {
            "credentials": info
        }
        return lambda: instance

    scan = block_coverage.blocks_by_provider
    try:
        mocker.patch.object(
            block_coverage, "load_all_blocks", return_value={"on": block(False)}
        )
        scan.cache_clear()
        assert scan()["acme_legacy"].auth_types == frozenset({"api_key"})
        mocker.patch.object(
            block_coverage, "load_all_blocks", return_value={"off": block(True)}
        )
        scan.cache_clear()
        # The disabled block still says what Acme covers, not how to sign in.
        assert scan()["acme_legacy"].auth_types == frozenset()
        assert scan()["acme_legacy"].products == frozenset({"acme", "send"})
    finally:
        scan.cache_clear()


def test_a_provider_switching_field_credits_each_provider_its_own_types():
    """Codex's block takes an OpenAI key or a ChatGPT sign-in through one
    field: Codex gains the sign-in, not the key."""
    blocks = block_coverage.blocks_by_provider()

    assert blocks["codex"].auth_types == frozenset({"oauth2"})
    assert "api_key" in blocks["openai"].auth_types


def test_a_device_code_sign_in_counts_without_an_oauth_app(mocker: MockerFixture):
    """Codex has no OAuth app, only a device-code handler, and that sign-in
    gives the user the OAuth account its blocks take."""
    mocker.patch.object(support, "oauth_client_credentials", return_value=None)

    found = support.find_connectable("codex")

    assert found is not None and found.key == "codex"


GOOGLE = ConnectableService(
    key="google",
    name="Google",
    products=frozenset({"google", "sheet", "doc", "drive", "calendar", "gmail"}),
    product_names=frozenset({"gmail", "drive", "calendar", "sheets"}),
)
NOTION = ConnectableService(
    key="notion", name="Notion", products=frozenset({"notion", "page", "database"})
)
CANVA_MCP = ConnectableService(key="canva", name="Canva")  # no blocks


@pytest.mark.parametrize(
    "name, key",
    [
        ("Google Sheets", "google"),
        ("google_sheets", "google"),
        ("Google Drive", "google"),
        ("Google Sheets API", "google"),
        ("Gmail", "google"),  # a product only Google's blocks lead with
        ("Notion", "notion"),
        ("Canva designs", "canva"),  # a server is the vendor's whole surface
        ("Google Analytics", None),
        ("Google Ads", None),
        ("Notion Calendar", None),
        ("Customer support", None),
    ],
)
def test_a_vendors_name_counts_only_with_products_its_blocks_cover(
    mocker: MockerFixture, name: str, key: str | None
):
    """ "Google Analytics" is not Google Sheets: the words after a vendor's
    name must be products its blocks cover."""
    mocker.patch.object(
        support, "connectable_services", return_value=[GOOGLE, NOTION, CANVA_MCP]
    )

    found = support.find_connectable(name)

    assert (found.key if found else None) == key


def test_the_live_registry_knows_what_googles_blocks_cover(mocker: MockerFixture):
    mocker.patch.object(support, "oauth_client_credentials", return_value=("i", "s"))

    assert support.find_connectable("Google Sheets").key == "google"
    assert support.find_connectable("Gmail").key == "google"
    assert support.find_connectable("Google Analytics") is None
    assert support.find_connectable("Slack Huddles") is None
    # Block names lead with verbs too (BanSubredditUserBlock): not products.
    assert support.find_connectable("Ban") is None
    assert support.find_connectable("Validate") is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Thank each new follower on Twitter", ["Twitter"]),
        ("Log each order in Google Sheets", ["Google"]),
        ("Post the weekly update to Reddit", ["Reddit"]),
        ("Open an issue on GitHub for each bug", ["GitHub"]),  # as written
        ("Work out the notion of a perfect sticker", []),  # a word, not Notion
        ("Draft the post in Printify's voice", []),  # connectable here
    ],
)
def test_a_task_naming_a_service_we_know_but_cannot_connect_is_labelled(
    mocker: MockerFixture, text: str, expected: list[str]
):
    """A task that declares no integrations ([]) still says it needs a
    service the platform knows (a provider, a product its blocks cover, an
    MCP server) when this deployment cannot connect it: Google without its
    OAuth app, Reddit, GitHub, Notion."""
    mocker.patch.object(support, "oauth_client_credentials", return_value=None)
    services = [
        s
        for s in support.connectable_services()
        if s.key not in {"github", "notion", "reddit"}
    ]

    assert support.unsupported_in_task([], text, services) == expected
