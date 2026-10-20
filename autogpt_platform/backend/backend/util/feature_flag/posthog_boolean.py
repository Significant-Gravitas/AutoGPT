from typing import Any

from posthog import Posthog
from posthog.types import FeatureFlagResult


def evaluate_boolean(
    client: Posthog,
    flag_key: str,
    distinct_id: str,
    person_properties: dict[str, Any] | None,
    default: bool,
) -> tuple[Any, bool]:
    """Keep a complete remote omission separate from an unavailable evaluation."""
    result = (
        client.get_feature_flag_result(
            flag_key,
            distinct_id,
            person_properties=person_properties or None,
            only_evaluate_locally=True,
            send_feature_flag_events=False,
        )
        if client.enable_local_evaluation or client.feature_flags
        else None
    )
    if result is not None:
        if not isinstance(result.enabled, bool):
            return default, False
        if result.enabled is False:
            return False, True
        return (
            result.payload if result.payload is not None else result.get_value()
        ), True

    response = client.get_flags_decision(
        distinct_id,
        person_properties=person_properties or None,
        flag_keys_to_evaluate=[flag_key],
    )
    # The SDK normalizes an HTTP failure to an empty response without this field.
    # A complete response omits inactive and nonexistent flags alike.
    detail = response.get("flags", {}).get(flag_key)
    if detail is None:
        if response.get("errorsWhileComputingFlags") is False:
            return False, True
        return default, False
    if not isinstance(detail.enabled, bool):
        return default, False
    if detail.enabled is False:
        return False, True

    result = FeatureFlagResult.from_flag_details(detail)
    if result is None:
        return default, False
    return result.payload if result.payload is not None else result.get_value(), True
