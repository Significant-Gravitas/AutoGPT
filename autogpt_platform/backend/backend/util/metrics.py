import sentry_sdk

from backend.util.settings import Settings


def sentry_init():
    sentry_dsn = Settings().secrets.sentry_dsn
    sentry_sdk.init(dsn=sentry_dsn, traces_sample_rate=1.0, profiles_sample_rate=1.0)
