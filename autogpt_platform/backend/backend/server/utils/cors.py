from __future__ import annotations

from typing import Dict, List, Sequence

from backend.util.settings import AppEnvironment


def build_cors_params(
    origins: Sequence[str], app_env: AppEnvironment
) -> Dict[str, object]:
    allow_origins: List[str] = []
    regex_patterns: List[str] = []

    if app_env == AppEnvironment.PRODUCTION:
        for origin in origins:
            if origin.startswith("regex:"):
                pattern = origin[len("regex:") :].lower()
                if "localhost" in pattern or "127.0.0.1" in pattern:
                    raise ValueError(
                        "Production environment cannot allow localhost origins via regex"
                    )
                continue

            lowered = origin.lower()
            if "localhost" in lowered or "127.0.0.1" in lowered:
                raise ValueError(
                    "Production environment cannot allow localhost origins"
                )

    for origin in origins:
        if origin.startswith("regex:"):
            regex_patterns.append(origin[len("regex:") :])
        else:
            allow_origins.append(origin)

    allow_origin_regex = None
    if regex_patterns:
        combined_pattern = "|".join(f"(?:{pattern})" for pattern in regex_patterns)
        allow_origin_regex = f"^(?:{combined_pattern})$"

    return {
        "allow_origins": allow_origins,
        "allow_origin_regex": allow_origin_regex,
    }

