import os

import click

from .utils import env_file_exists, get_git_user_email, set_env_config_value


def setup_telemetry() -> None:
    if os.getenv("TELEMETRY_OPT_IN") is None:
        # If no .env file is present, don't bother asking to enable telemetry,
        # to prevent repeated asking in non-persistent environments.
        if not env_file_exists():
            return

        print()
        allow_telemetry = click.prompt(
            "❓ Do you want to enable telemetry? ❓\n"
            "This means AutoGPT will send diagnostic data to the core development team "
            "when something goes wrong,\nand will help us to diagnose and fix problems "
            "earlier and faster.\n\n"
            "By entering 'yes', you agree that you have read and agree to our Privacy"
            " Policy available here:\n"
            "https://www.notion.so/auto-gpt/Privacy-Policy-ab11c9c20dbd4de1a15dcffe84d77984"
            "\n\nPlease enter 'yes' or 'no'",
            type=bool,
        )
        set_env_config_value("TELEMETRY_OPT_IN", "true" if allow_telemetry else "false")
        print(
            "💡 If you ever change your mind, you can adjust 'TELEMETRY_OPT_IN' in .env"
        )
        print()

    if os.getenv("TELEMETRY_OPT_IN", "").lower() == "true":
        _setup_sentry()


def _setup_sentry() -> None:
    import sentry_sdk

    sentry_sdk.init(
        dsn="https://dc266f2f7a2381194d1c0fa36dff67d8@o4505260022104064.ingest.sentry.io/4506739844710400",  # noqa
        environment=os.getenv("TELEMETRY_ENVIRONMENT"),
    )

    # Allow Sentry to distinguish between users
    sentry_sdk.set_user({"email": get_git_user_email(), "ip_address": "{{auto}}"})
