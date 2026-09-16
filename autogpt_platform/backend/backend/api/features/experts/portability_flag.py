"""The gate on every expert-portability route.

Fail-closed and 404 rather than 403: with the flag off the feature does not
exist, so a URL someone guessed must not confirm that it will one day.

Deliberately not ``create_feature_flag_dependency``: that helper 404s whenever
LaunchDarkly has no SDK key, before consulting the ``FORCE_FLAG_EXPERT_PORTABILITY``
override every local environment relies on.
"""

import autogpt_libs.auth
import fastapi
from fastapi import Security

from backend.util.feature_flag import Flag, is_feature_enabled


async def require_expert_portability_flag(
    user_id: str | None = Security(autogpt_libs.auth.get_optional_user_id),
) -> None:
    if not await is_feature_enabled(Flag.EXPERT_PORTABILITY, user_id or "anonymous"):
        raise fastapi.HTTPException(status_code=404, detail="Feature not available")
