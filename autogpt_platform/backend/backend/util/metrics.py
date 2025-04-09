import logging

import sentry_sdk
from sentry_sdk.integrations.anthropic import AnthropicIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

from backend.util.settings import Settings


def sentry_init():
    sentry_dsn = Settings().secrets.sentry_dsn
    sentry_sdk.init(
        dsn=sentry_dsn,
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
        environment=f"app:{Settings().config.app_env.value}-behave:{Settings().config.behave_as.value}",
        _experiments={
            "enable_logs": True,
        },
        integrations=[
            LoggingIntegration(sentry_logs_level=logging.INFO),
            AnthropicIntegration(
                include_prompts=False,
            ),
        ],
    )
