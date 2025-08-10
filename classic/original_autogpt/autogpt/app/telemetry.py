import os

import click
from colorama import Fore, Style

from .utils import (
    env_file_exists,
    get_git_user_email,
    set_env_config_value,
    vcs_state_diverges_from_master,
)


def setup_telemetry() -> None:
    if os.getenv("TELEMETRY_OPT_IN") is None:
        # If no .env file is present, don't bother asking to enable telemetry,
        # to prevent repeated asking in non-persistent environments.
        if not env_file_exists():
            return

        allow_telemetry = click.prompt(
            f"""
{Style.BRIGHT}❓ Do you want to enable telemetry? ❓{Style.NORMAL}
This means AutoGPT will send diagnostic data to the core development team when something
goes wrong, and will help us to diagnose and fix problems earlier and faster. It also
allows us to collect basic performance data, which helps us find bottlenecks and other
things that slow down the application.

By entering 'yes', you confirm that you have read and agree to our Privacy Policy,
which is available here:
https://www.notion.so/auto-gpt/Privacy-Policy-ab11c9c20dbd4de1a15dcffe84d77984

Please enter 'yes' or 'no'""",
            type=bool,
        )
        set_env_config_value("TELEMETRY_OPT_IN", "true" if allow_telemetry else "false")
        click.echo(
            f"❤️  Thank you! Telemetry is {Fore.GREEN}enabled{Fore.RESET}."
            if allow_telemetry
            else f"👍 Telemetry is {Fore.RED}disabled{Fore.RESET}."
        )
        click.echo(
            "💡 If you ever change your mind, you can change 'TELEMETRY_OPT_IN' in .env"
        )
        click.echo()

    if os.getenv("TELEMETRY_OPT_IN", "").lower() == "true":
        _setup_sentry()


def _setup_sentry() -> None:
    import sentry_sdk

    sentry_sdk.init(
        dsn="https://dc266f2f7a2381194d1c0fa36dff67d8@o4505260022104064.ingest.sentry.io/4506739844710400",  # noqa
        enable_tracing=True,
        environment=os.getenv(
            "TELEMETRY_ENVIRONMENT",
            "production" if not vcs_state_diverges_from_master() else "dev",
        ),
    )

    # Allow Sentry to distinguish between users
    sentry_sdk.set_user({"email": get_git_user_email(), "ip_address": "{{auto}}"})
