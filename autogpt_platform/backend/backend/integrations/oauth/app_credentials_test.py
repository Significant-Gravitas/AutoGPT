"""Whether a provider's OAuth app is configured in this deployment."""

from pytest_mock import MockerFixture

from backend.integrations.oauth import app_credentials


def test_a_provider_with_its_app_configured_has_credentials(mocker: MockerFixture):
    secrets = mocker.patch.object(app_credentials.settings, "secrets")
    secrets.google_client_id = "id"
    secrets.google_client_secret = "secret"

    assert app_credentials.oauth_client_credentials("google") == ("id", "secret")


def test_a_declared_oauth_provider_without_its_app_has_none(mocker: MockerFixture):
    """Google and Reddit declare OAuth whether or not their apps are set up."""
    secrets = mocker.patch.object(app_credentials.settings, "secrets")
    secrets.google_client_id = ""
    secrets.google_client_secret = ""
    secrets.reddit_client_id = "id"
    secrets.reddit_client_secret = ""

    assert app_credentials.oauth_client_credentials("google") is None
    assert app_credentials.oauth_client_credentials("reddit") is None


def test_a_provider_with_no_oauth_handler_has_none():
    assert app_credentials.oauth_client_credentials("printify") is None
