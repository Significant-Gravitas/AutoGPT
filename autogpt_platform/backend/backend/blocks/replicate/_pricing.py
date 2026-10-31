"""Replicate's published prices for its official models.

Official models are billed per output (an image, a second of video, a token)
rather than by run time, so the block prices their runs from this table.
Prices are from each model's pricing section on replicate.com, read on
2026-10-30.
"""

import math
from collections.abc import Mapping, Sequence

from pydantic import BaseModel

InputValue = str | int | float | bool


class Rate(BaseModel):
    usd: float
    when: dict[str, InputValue] = {}


class MetricPrice(BaseModel):
    """The price of one quantity Replicate bills, named as in a run's metrics."""

    metric: str
    # First match wins; the last is unconditional and the dearest, so an input
    # value the table does not know is never charged at a cheaper tier.
    rates: list[Rate]
    defaults: dict[str, InputValue] = {}
    units_from: tuple[str, ...] = ()
    max_units: float | None = None


def official_run_cost_usd(
    prices: Sequence[MetricPrice],
    model_inputs: Mapping[str, object],
    metrics: Mapping[str, object],
    output: object,
) -> float:
    """What Replicate charges for one run of an official model, in USD."""
    return sum(
        _rate(price, model_inputs) * _units(price, model_inputs, metrics, output)
        for price in prices
    )


def _rate(price: MetricPrice, model_inputs: Mapping[str, object]) -> float:
    return next(
        rate.usd
        for rate in price.rates
        if all(
            model_inputs.get(key, price.defaults.get(key)) == value
            for key, value in rate.when.items()
        )
    )


def _units(
    price: MetricPrice,
    model_inputs: Mapping[str, object],
    metrics: Mapping[str, object],
    output: object,
) -> float:
    reported = _reported_units(price.metric, metrics)
    if reported is not None:
        return reported
    if price.metric.endswith("_output_count"):
        return float(len(output)) if isinstance(output, list) else 1.0
    from_inputs = sum(
        _input_units(model_inputs.get(key, price.defaults.get(key)))
        for key in price.units_from
    )
    if from_inputs > 0:
        return from_inputs
    return price.max_units or 0.0


# Older models report these names instead of the billing metric's own.
_METRIC_ALIASES = {
    "image_output_count": "image_count",
    "token_input_count": "input_token_count",
    "token_output_count": "output_token_count",
}


def _reported_units(metric: str, metrics: Mapping[str, object]) -> float | None:
    for key in (f"billing.{metric}", metric, _METRIC_ALIASES.get(metric)):
        value = metrics.get(key) if key else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _input_units(value: object) -> float:
    """A number counts as itself; text counts its characters, no fewer than its tokens."""
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else 0.0
    text = str(value)
    try:
        number = float(text)
    except ValueError:
        return float(len(text))
    return number if math.isfinite(number) else float(len(text))


_IMAGES = "image_output_count"
_VIDEO_SECONDS = "video_output_duration_seconds"
_PER_K = 1e-3
_PER_M = 1e-6


def _price(
    metric: str,
    top_usd: float,
    *tiers: tuple[dict[str, InputValue], float],
    defaults: dict[str, InputValue] | None = None,
    units_from: tuple[str, ...] = (),
    max_units: float | None = None,
) -> MetricPrice:
    return MetricPrice(
        metric=metric,
        rates=[Rate(when=when, usd=usd) for when, usd in tiers] + [Rate(usd=top_usd)],
        defaults=defaults or {},
        units_from=units_from,
        max_units=max_units,
    )


def _per_second(
    top_usd: float,
    *tiers: tuple[dict[str, InputValue], float],
    defaults: dict[str, InputValue],
    duration: str | None = "duration",
    max_seconds: float | None = None,
) -> tuple[MetricPrice, ...]:
    return (
        _price(
            _VIDEO_SECONDS,
            top_usd,
            *tiers,
            defaults=defaults,
            units_from=(duration,) if duration else (),
            max_units=max_seconds,
        ),
    )


def _per_image(
    top_usd: float,
    *tiers: tuple[dict[str, InputValue], float],
    defaults: dict[str, InputValue] | None = None,
) -> tuple[MetricPrice, ...]:
    return (_price(_IMAGES, top_usd, *tiers, defaults=defaults),)


def _per_token(
    input_usd: float, output_usd: float, *, max_tokens: int
) -> tuple[MetricPrice, ...]:
    return (
        _price("token_input_count", input_usd, units_from=("prompt", "system_prompt")),
        _price(
            "token_output_count",
            output_usd,
            defaults={"max_tokens": max_tokens},
            units_from=("max_tokens", "max_new_tokens"),
        ),
    )


def _per_input_text(usd: float, metric: str, text: str) -> tuple[MetricPrice, ...]:
    return (_price(metric, usd, units_from=(text,)),)


_NO_AUDIO: dict[str, InputValue] = {"generate_audio": False}
_VEO_DEFAULTS: dict[str, InputValue] = {"generate_audio": True, "duration": 8}

OFFICIAL_MODEL_PRICES: dict[str, tuple[MetricPrice, ...]] = {
    # Video, per second of output
    "google/veo-3.1": _per_second(0.40, (_NO_AUDIO, 0.20), defaults=_VEO_DEFAULTS),
    "google/veo-3.1-fast": _per_second(0.15, (_NO_AUDIO, 0.10), defaults=_VEO_DEFAULTS),
    "google/veo-3": _per_second(0.40, (_NO_AUDIO, 0.20), defaults=_VEO_DEFAULTS),
    "google/veo-3-fast": _per_second(0.15, (_NO_AUDIO, 0.10), defaults=_VEO_DEFAULTS),
    "google/veo-3.1-lite": _per_second(
        0.08,
        ({"resolution": "720p"}, 0.05),
        defaults={"resolution": "720p", "duration": 8},
    ),
    "prunaai/p-video": _per_second(
        0.04,
        ({"draft": True, "resolution": "720p"}, 0.005),
        ({"draft": True, "resolution": "1080p"}, 0.01),
        ({"draft": False, "resolution": "720p"}, 0.02),
        defaults={"draft": False, "resolution": "720p", "duration": 5},
    ),
    "prunaai/p-video-2": _per_second(
        0.06,
        ({"draft": True, "resolution": "720p"}, 0.015),
        ({"draft": True, "resolution": "1080p"}, 0.03),
        ({"draft": False, "resolution": "720p"}, 0.03),
        defaults={"draft": False, "resolution": "720p"},
        max_seconds=20,
    ),
    # Its length follows the audio or script, so no input bounds it.
    "prunaai/p-video-avatar": _per_second(
        0.045,
        ({"resolution": "720p"}, 0.025),
        defaults={"resolution": "720p"},
        duration=None,
        max_seconds=60,
    ),
    "kwaivgi/kling-v2.1": _per_second(
        0.09,
        ({"mode": "standard"}, 0.05),
        defaults={"mode": "standard", "duration": 5},
    ),
    "kwaivgi/kling-v2.5-turbo-pro": _per_second(0.07, defaults={"duration": 5}),
    "kwaivgi/kling-v3-video": _per_second(
        0.42,
        ({"mode": "standard", "generate_audio": False}, 0.168),
        ({"mode": "standard", "generate_audio": True}, 0.252),
        ({"mode": "pro", "generate_audio": False}, 0.224),
        ({"mode": "pro", "generate_audio": True}, 0.336),
        defaults={"mode": "pro", "generate_audio": False, "duration": 5},
    ),
    "bytedance/seedance-1-lite": _per_second(
        0.072,
        ({"resolution": "480p"}, 0.018),
        ({"resolution": "720p"}, 0.036),
        defaults={"resolution": "720p", "duration": 5},
    ),
    "bytedance/seedance-1-pro": _per_second(
        0.15,
        ({"resolution": "480p"}, 0.03),
        ({"resolution": "720p"}, 0.06),
        defaults={"resolution": "1080p", "duration": 5},
    ),
    # Priced at the video-input rates; a duration of -1 lets the model choose.
    "bytedance/seedance-2.0": _per_second(
        1.25,
        ({"resolution": "480p"}, 0.10),
        ({"resolution": "720p"}, 0.22),
        ({"resolution": "1080p"}, 0.55),
        defaults={"resolution": "720p", "duration": 5},
        max_seconds=15,
    ),
    "wan-video/wan-2.5-t2v": _per_second(
        0.15,
        ({"size": "832*480"}, 0.05),
        ({"size": "480*832"}, 0.05),
        ({"size": "1280*720"}, 0.10),
        ({"size": "720*1280"}, 0.10),
        defaults={"size": "1280*720", "duration": 5},
    ),
    "wan-video/wan-2.5-i2v": _per_second(
        0.15,
        ({"resolution": "480p"}, 0.05),
        ({"resolution": "720p"}, 0.10),
        defaults={"resolution": "720p", "duration": 5},
    ),
    "lightricks/ltx-2-fast": _per_second(
        0.16,
        ({"resolution": "1080p"}, 0.04),
        ({"resolution": "2k"}, 0.08),
        defaults={"resolution": "1080p", "duration": 6},
    ),
    "lightricks/ltx-2-pro": _per_second(
        0.24,
        ({"resolution": "1080p"}, 0.06),
        ({"resolution": "2k"}, 0.12),
        defaults={"resolution": "1080p", "duration": 6},
    ),
    "openai/sora-2": _per_second(0.10, defaults={"seconds": 4}, duration="seconds"),
    "openai/sora-2-pro": _per_second(
        0.50,
        ({"resolution": "standard"}, 0.30),
        defaults={"resolution": "standard", "seconds": 4},
        duration="seconds",
    ),
    "runwayml/gen-4.5": _per_second(0.12, defaults={"duration": 5}),
    "xai/grok-imagine-video": _per_second(0.05, defaults={"duration": 5}),
    # Video, per output video
    "minimax/hailuo-02": (
        _price(
            "video_output_count",
            0.48,
            ({"resolution": "512p", "duration": 6}, 0.10),
            ({"resolution": "512p", "duration": 10}, 0.15),
            ({"resolution": "768p", "duration": 6}, 0.27),
            ({"resolution": "768p", "duration": 10}, 0.45),
            defaults={"resolution": "1080p", "duration": 6},
        ),
    ),
    # Images, per output image
    "black-forest-labs/flux-schnell": _per_image(0.003),
    "black-forest-labs/flux-dev": _per_image(0.025),
    "black-forest-labs/flux-1.1-pro": _per_image(0.04),
    "black-forest-labs/flux-1.1-pro-ultra": _per_image(0.06),
    "black-forest-labs/flux-kontext-pro": _per_image(0.04),
    "black-forest-labs/flux-kontext-max": _per_image(0.08),
    "google/nano-banana": _per_image(0.039),
    "google/nano-banana-2": _per_image(
        0.151,
        ({"resolution": "1K"}, 0.067),
        ({"resolution": "2K"}, 0.101),
        defaults={"resolution": "1K"},
    ),
    "google/nano-banana-pro": _per_image(
        0.30,
        ({"resolution": "1K"}, 0.15),
        ({"resolution": "2K"}, 0.15),
        defaults={"resolution": "2K"},
    ),
    "google/imagen-4": _per_image(0.04),
    "google/imagen-4-fast": _per_image(0.02),
    "google/imagen-4-ultra": _per_image(0.06),
    "ideogram-ai/ideogram-v3-turbo": _per_image(0.03),
    "ideogram-ai/ideogram-v3-quality": _per_image(0.09),
    "recraft-ai/recraft-v3": _per_image(0.04),
    "bytedance/seedream-4": _per_image(0.03),
    "qwen/qwen-image": _per_image(0.025),
    "prunaai/p-image": _per_image(0.005),
    # Audio
    "minimax/speech-2.6-hd": _per_input_text(
        0.10 * _PER_K, "token_input_count", "text"
    ),
    "minimax/speech-2.6-turbo": _per_input_text(
        0.06 * _PER_K, "token_input_count", "text"
    ),
    "minimax/speech-02-hd": _per_input_text(0.10 * _PER_K, "token_input_count", "text"),
    "minimax/speech-02-turbo": _per_input_text(
        0.06 * _PER_K, "token_input_count", "text"
    ),
    "elevenlabs/v3": _per_input_text(0.10 * _PER_K, "character_input_count", "prompt"),
    "minimax/music-1.5": (_price("audio_output_count", 0.03),),
    # Language models, per token
    "meta/meta-llama-3-8b-instruct": _per_token(
        0.05 * _PER_M, 0.25 * _PER_M, max_tokens=512
    ),
    "meta/meta-llama-3-70b-instruct": _per_token(
        0.65 * _PER_M, 2.75 * _PER_M, max_tokens=512
    ),
    "meta/llama-2-7b-chat": _per_token(0.05 * _PER_M, 0.25 * _PER_M, max_tokens=512),
    "openai/gpt-oss-120b": _per_token(0.18 * _PER_M, 0.72 * _PER_M, max_tokens=2048),
    "deepseek-ai/deepseek-r1": _per_token(
        3.75 * _PER_M, 0.01 * _PER_K, max_tokens=2048
    ),
}
