"""Credential-connection failures that Sentry would otherwise never see.

`sentry_init` builds `LoggingIntegration()` with sentry-sdk's default
`event_level=ERROR`, so a `logger.warning` on these paths raises no event and
reaches nobody. The paths below are the ones where nothing else reports the
failure — the user sees a stuck card and we see nothing at all.

Class numbers come from the credential-connection failure map; the frontend
tags the same numbers on its own terminal-state events.
"""

import logging
from enum import Enum

import sentry_sdk


class CredentialFailure(str, Enum):
    """Failure classes an alert rule groups on. Renaming one breaks that rule.

    Only the classes this service can observe. Classes 5, 11 and 13 are
    client-side terminal states and are named in the frontend's own map.
    """

    PROVIDER_UNKNOWN_TO_FRONTEND = "class_03_provider_unknown_to_frontend"
    PROVIDER_REGISTRATION_WRONG = "class_06_provider_registration_wrong"
    DEVICE_CODE_RACE = "class_07_device_code_race"
    SCOPES_TOO_NARROW = "class_08_scopes_too_narrow"
    MANAGED_PROVISIONING_LATE = "class_12_managed_provisioning_late"


def report_credential_failure(
    logger: logging.Logger,
    failure_class: CredentialFailure,
    reason: str,
    message: str,
    *,
    provider: str | None = None,
    **context: object,
) -> None:
    """Log at ERROR so the Sentry event exists, tagged so a rule can find it.

    Only `failure_class`, `reason` and `provider` become tags — anything
    per-user or per-request stays an extra, because Sentry tags are indexed
    and high-cardinality values there are what makes a project unqueryable.
    `json_fields` is the only key GCP's StructuredLogHandler carries through,
    so the same fields are queryable in Cloud Logging.
    """
    tags = {"failure_class": failure_class.value, "reason": reason}
    if provider is not None:
        tags["provider"] = provider
    fields = {**tags, **context}

    with sentry_sdk.new_scope() as scope:
        for tag, value in tags.items():
            scope.set_tag(tag, value)
        logger.error(message, extra={**fields, "json_fields": fields})
