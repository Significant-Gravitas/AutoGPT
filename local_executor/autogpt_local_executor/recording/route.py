"""
Interpretation-route probing — detect local capability, only then ask (§3.1).

On START_RECORDING the shim probes, in order:

  1. a11y / DOM already captured for the demonstrated steps → free structured
     text → `extract_then_cloud` works.
  2. OCR available on-device → `extract_then_cloud` works even for `kind: none`
     (canvas/Electron) steps.
  3. Capable local vision model present → offer `local_vlm` (pixels stay local).
  4. None of the above → `screenshots_to_cloud`, the ONLY route that prompts
     the user (§9.1).

Each probe is a small checker. The VLM/OCR probes are capability checks — is
the dependency importable, is a model listed — they do NOT actually run a
model. Matching the prompt to the actual incremental risk is deliberate (§3.1):
only `screenshots_to_cloud` requires consent.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

from ..protocol import InterpretationRoute, RecordingChannel

logger = logging.getLogger(__name__)

# llava-class and friends — substrings that mark a model as vision-capable
# enough to author a skill on-device (§3 `local_vlm` "llava-class+").
_VLM_NAME_HINTS = ("llava", "bakllava", "llama3.2-vision", "qwen2-vl", "minicpm-v", "moondream")


@dataclass(frozen=True)
class RouteDecision:
    """Result of probing. `requires_consent` is true ONLY for
    screenshots_to_cloud (§3.1) — the other routes keep pixels local. Every
    route may still send the hygiene-redacted structured trajectory for
    authenticated browser review."""

    route: InterpretationRoute
    requires_consent: bool
    reason: str


def _structured_channels_present(channels: Iterable[str]) -> bool:
    """Probe 1: a11y/DOM channels are requested → structured text is free.

    We treat the *requested* channels as the signal here: if the platform asked
    for browser or desktop_ax enrichment, those channels resolve structured
    text per step and extract_then_cloud is lossless/cheap. The floor alone is
    pixels-only and doesn't satisfy this probe. Accepts plain strings so the
    HELLO-time route advertisement (config channels are str) can reuse it.
    """
    chans = set(channels)
    return bool(chans & {"browser", "desktop_ax"})


def _ocr_available() -> bool:
    """Probe 2: an on-device OCR engine is importable.

    Capability check only — we never run OCR here. Bundleable engines first
    (their presence is the cheap happy path), then common system ones.
    """
    for mod in ("rapidocr_onnxruntime", "easyocr", "pytesseract"):
        if _module_importable(mod):
            return True
    return False


def _local_vlm_present(local_llm_models: Iterable[str]) -> bool:
    """Probe 3: a capable local vision model is listed.

    We do NOT load or run the model — we just check the advertised model list
    (the same list HELLO carries from the Ollama probe) for a vision-capable
    name. This is a name-based capability check by design.
    """
    for name in local_llm_models:
        lname = name.lower()
        if any(hint in lname for hint in _VLM_NAME_HINTS):
            return True
    return False


def _module_importable(name: str) -> bool:
    """True if `name` can be imported without actually importing it (cheap)."""
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def probe_interpretation_route(
    *,
    channels: Iterable[RecordingChannel],
    local_llm_models: Iterable[str] = (),
    requested: InterpretationRoute | None = None,
) -> RouteDecision:
    """Choose the interpretation route by probing local capability (§3.1).

    `requested` honors an explicit platform choice when the machine can support
    it; otherwise we fall through the probe order. The returned
    `requires_consent` flag is true only for screenshots_to_cloud.

    Probe precedence when no explicit (supportable) request is given:
      structured channels → OCR → local VLM → screenshots_to_cloud.

    Note the doc lists local_vlm as the per-machine default when a VLM is
    present (§3.1 step 3), but only *after* the cheaper extract_then_cloud
    probes fail — extract_then_cloud generalizes with cloud-grade reasoning and
    is the documented default (§3). We therefore prefer extract_then_cloud when
    structured text or OCR is available, and offer local_vlm only when neither
    is. An explicit `requested=local_vlm` is honored whenever a VLM is present.
    """
    channels = list(channels)
    local_llm_models = list(local_llm_models)

    # Honor an explicit, supportable request first.
    if requested == "local_vlm" and _local_vlm_present(local_llm_models):
        return RouteDecision(
            route="local_vlm",
            requires_consent=False,
            reason="explicit local_vlm honored; capable local vision model present",
        )
    if requested == "extract_then_cloud" and (
        _structured_channels_present(channels) or _ocr_available()
    ):
        return RouteDecision(
            route="extract_then_cloud",
            requires_consent=False,
            reason="explicit extract_then_cloud honored; structured text or OCR available",
        )
    if requested == "screenshots_to_cloud":
        # The platform explicitly chose the cloud fallback. Still consent-gated.
        return RouteDecision(
            route="screenshots_to_cloud",
            requires_consent=True,
            reason="explicit screenshots_to_cloud; consent required (§9.1)",
        )

    # Probe order (§3.1).
    if _structured_channels_present(channels):
        return RouteDecision(
            route="extract_then_cloud",
            requires_consent=False,
            reason="a11y/DOM channels requested → structured text is free",
        )
    if _ocr_available():
        return RouteDecision(
            route="extract_then_cloud",
            requires_consent=False,
            reason="on-device OCR available → extract_then_cloud covers kind:none",
        )
    if _local_vlm_present(local_llm_models):
        return RouteDecision(
            route="local_vlm",
            requires_consent=False,
            reason="capable local vision model present → pixel-local local_vlm",
        )
    return RouteDecision(
        route="screenshots_to_cloud",
        requires_consent=True,
        reason="no local extractor or VLM → cloud fallback, consent required (§9.1)",
    )


__all__ = [
    "RouteDecision",
    "probe_interpretation_route",
]
