"""Admin write operations for LLM routing cells (LlmModelRoute)."""

from __future__ import annotations

import logging

import prisma.models

logger = logging.getLogger(__name__)


class UnknownRouteModelError(LookupError):
    """The routing cell references a slug the registry doesn't know."""


async def list_routes() -> list[prisma.models.LlmModelRoute]:
    return await prisma.models.LlmModelRoute.prisma().find_many(
        order=[{"surface": "asc"}, {"mode": "asc"}, {"tier": "asc"}]
    )


def _capability_warnings(model: prisma.models.LlmModel, mode: str) -> list[str]:
    warnings: list[str] = []
    if mode == "thinking" and not model.supportsReasoning:
        warnings.append(
            f"model '{model.slug}' does not advertise reasoning support but is "
            f"being routed for a thinking cell"
        )
    if not model.supportsTools:
        warnings.append(f"model '{model.slug}' does not advertise tool support")
    return warnings


async def set_route(
    surface: str, mode: str, tier: str, model_slug: str | None
) -> tuple[prisma.models.LlmModelRoute | None, list[str]]:
    """Upsert (or delete, when model_slug is None) a routing cell.

    The cell model must exist and be enabled (the kill switch beats routing);
    HIDDEN visibility is allowed — that's the pre-launch testing state.
    Returns the row (None after delete) and capability warnings for the admin
    UI. Caller owns audit + cache refresh.
    """
    if model_slug is None:
        await prisma.models.LlmModelRoute.prisma().delete_many(
            where={"surface": surface, "mode": mode, "tier": tier}
        )
        return None, []

    model = await prisma.models.LlmModel.prisma().find_unique(
        where={"slug": model_slug}
    )
    if model is None:
        raise UnknownRouteModelError(
            f"Model '{model_slug}' does not exist in the registry"
        )
    if not model.isEnabled:
        raise ValueError(
            f"Model '{model_slug}' is disabled (kill switch) and cannot be routed"
        )

    row = await prisma.models.LlmModelRoute.prisma().upsert(
        where={"surface_mode_tier": {"surface": surface, "mode": mode, "tier": tier}},
        data={
            "create": {
                "surface": surface,
                "mode": mode,
                "tier": tier,
                "modelSlug": model_slug,
            },
            "update": {"modelSlug": model_slug},
        },
    )
    return row, _capability_warnings(model, mode)
