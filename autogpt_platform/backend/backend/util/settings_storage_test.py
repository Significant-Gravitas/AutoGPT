import pytest

from backend.util.settings import Config, _warn_single_bucket


@pytest.fixture(autouse=True)
def no_bucket_env(monkeypatch):
    # Config reads the environment before keyword arguments, and backend.data.db
    # copies .env into os.environ on import.
    for name in (
        "BEHAVE_AS",
        "MEDIA_GCS_BUCKET_NAME",
        "PUBLIC_SITE_MEDIA_BUCKET",
        "PRIVATE_USER_DATA_BUCKET",
    ):
        monkeypatch.delenv(name, raising=False)


def test_storage_bucket_settings_use_distinct_public_and_private_names():
    config = Config(
        _env_file=None,
        PUBLIC_SITE_MEDIA_BUCKET="public-media",
        PRIVATE_USER_DATA_BUCKET="private-data",
    )

    assert config.resolved_public_site_media_bucket == "public-media"
    assert config.resolved_private_user_data_bucket == "private-data"


def test_legacy_media_bucket_env_remains_compatible_for_self_hosting():
    config = Config(
        _env_file=None,
        MEDIA_GCS_BUCKET_NAME="legacy-media",
    )

    assert config.resolved_public_site_media_bucket == "legacy-media"
    assert config.resolved_private_user_data_bucket == "legacy-media"


def test_public_media_env_takes_precedence_over_legacy_name():
    config = Config(
        _env_file=None,
        PUBLIC_SITE_MEDIA_BUCKET="public-media",
        PRIVATE_USER_DATA_BUCKET="private-data",
        MEDIA_GCS_BUCKET_NAME="legacy-media",
    )

    assert config.resolved_public_site_media_bucket == "public-media"
    assert config.resolved_private_user_data_bucket == "private-data"


def test_each_explicit_bucket_keeps_the_other_legacy_fallback():
    public_override = Config(
        _env_file=None,
        PUBLIC_SITE_MEDIA_BUCKET="public-media",
        MEDIA_GCS_BUCKET_NAME="legacy-media",
    )
    private_override = Config(
        _env_file=None,
        PRIVATE_USER_DATA_BUCKET="private-data",
        MEDIA_GCS_BUCKET_NAME="legacy-media",
    )

    assert public_override.resolved_public_site_media_bucket == "public-media"
    assert public_override.resolved_private_user_data_bucket == "legacy-media"
    assert private_override.resolved_public_site_media_bucket == "legacy-media"
    assert private_override.resolved_private_user_data_bucket == "private-data"


@pytest.mark.parametrize(
    "values",
    [
        {"PUBLIC_SITE_MEDIA_BUCKET": "public-media"},
        {"PRIVATE_USER_DATA_BUCKET": "private-data"},
        {
            "PUBLIC_SITE_MEDIA_BUCKET": "same-bucket",
            "PRIVATE_USER_DATA_BUCKET": "same-bucket",
        },
        {
            "PUBLIC_SITE_MEDIA_BUCKET": "legacy-media",
            "PRIVATE_USER_DATA_BUCKET": "private-data",
            "MEDIA_GCS_BUCKET_NAME": "legacy-media",
        },
        {
            "PUBLIC_SITE_MEDIA_BUCKET": "public-media",
            "PRIVATE_USER_DATA_BUCKET": "private-data",
            "MEDIA_GCS_BUCKET_NAME": "legacy-media",
        },
    ],
)
def test_cloud_split_configuration_fails_closed(values):
    with pytest.raises(ValueError):
        Config(_env_file=None, BEHAVE_AS="cloud", **values)


@pytest.mark.parametrize(
    "legacy_bucket,private_bucket",
    [("legacy-media", "legacy-media"), ("", "private-data")],
)
def test_cloud_split_configuration_accepts_safe_migration_states(
    legacy_bucket, private_bucket
):
    config = Config(
        _env_file=None,
        BEHAVE_AS="cloud",
        PUBLIC_SITE_MEDIA_BUCKET="public-media",
        PRIVATE_USER_DATA_BUCKET=private_bucket,
        MEDIA_GCS_BUCKET_NAME=legacy_bucket,
    )

    assert config.resolved_public_site_media_bucket == "public-media"
    assert config.resolved_private_user_data_bucket == private_bucket


@pytest.mark.parametrize("behave_as", ["cloud", "local"])
def test_a_deployment_on_the_legacy_bucket_warns_once(caplog, behave_as):
    _warn_single_bucket.cache_clear()

    for _ in range(2):
        Config(
            _env_file=None, BEHAVE_AS=behave_as, MEDIA_GCS_BUCKET_NAME="legacy-media"
        )

    warnings = [r for r in caplog.records if "PRIVATE_USER_DATA_BUCKET" in r.message]
    assert len(warnings) == 1
