import json

from backend.api.features.store.preview_seed import SeedSummary
from backend.util.settings import AppEnvironment

from . import seed_preview_marketplace as cli


async def test_production_is_refused_before_connecting(mocker, capsys):
    settings = mocker.patch("backend.api.features.store.preview_seed.Settings")
    settings.return_value.config.app_env = AppEnvironment.PRODUCTION
    settings.return_value.config.sentry_environment = ""
    connect = mocker.patch("backend.data.db.connect")

    assert await cli._run() == 2

    connect.assert_not_called()
    assert "refusing to seed synthetic data" in capsys.readouterr().err


async def test_a_clean_run_prints_its_summary_and_succeeds(mocker, capsys):
    mocker.patch.object(cli, "ensure_seedable")
    mocker.patch("backend.data.db.connect")
    disconnect = mocker.patch("backend.data.db.disconnect")
    mocker.patch.object(
        cli, "seed_marketplace", return_value=SeedSummary(users=9, reviews=58)
    )

    assert await cli._run() == 0

    disconnect.assert_awaited_once()
    assert json.loads(capsys.readouterr().out)["reviews"] == 58


async def test_a_listing_that_could_not_be_seeded_fails_the_run(mocker):
    mocker.patch.object(cli, "ensure_seedable")
    mocker.patch("backend.data.db.connect")
    mocker.patch("backend.data.db.disconnect")
    mocker.patch.object(
        cli,
        "seed_marketplace",
        return_value=SeedSummary(failed={"listing:utm-link-builder": "boom"}),
    )

    assert await cli._run() == 1
