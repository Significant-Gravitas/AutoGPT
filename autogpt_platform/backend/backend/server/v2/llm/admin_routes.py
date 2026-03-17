"""Admin write API for LLM registry management.

Provides endpoints for creating, updating, and deleting:
- Models
- Providers
- Costs
- Creators
- Migrations

All endpoints require admin authentication.
"""

from typing import Any

import autogpt_libs.auth
from fastapi import APIRouter, HTTPException, Security, status

from backend.server.v2.llm.admin_model import (
    CreateLlmModelRequest,
    CreateLlmProviderRequest,
    UpdateLlmModelRequest,
    UpdateLlmProviderRequest,
)

router = APIRouter()


@router.post(
    "/llm/models",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Security(autogpt_libs.auth.requires_admin_user)],
)
async def create_model(
    request: CreateLlmModelRequest,
) -> dict[str, Any]:
    """Create a new LLM model.

    Requires admin authentication.
    """
    # TODO: Implement model creation
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Model creation not yet implemented",
    )


@router.patch("/llm/models/{slug}", dependencies=[Security(autogpt_libs.auth.requires_admin_user)])
async def update_model(
    slug: str,
    request: UpdateLlmModelRequest,
) -> dict[str, Any]:
    """Update an existing LLM model.

    Requires admin authentication.
    """
    # TODO: Implement model update
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Model update not yet implemented",
    )


@router.delete("/llm/models/{slug}", dependencies=[Security(autogpt_libs.auth.requires_admin_user)], status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    slug: str,
) -> None:
    """Delete an LLM model.

    Requires admin authentication.
    """
    # TODO: Implement model deletion
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Model deletion not yet implemented",
    )


@router.post("/llm/providers", status_code=status.HTTP_201_CREATED)
async def create_provider(
    request: CreateLlmProviderRequest,
) -> dict[str, Any]:
    """Create a new LLM provider.

    Requires admin authentication.
    """
    # TODO: Implement provider creation
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Provider creation not yet implemented",
    )


@router.patch("/llm/providers/{name}", dependencies=[Security(autogpt_libs.auth.requires_admin_user)])
async def update_provider(
    name: str,
    request: UpdateLlmProviderRequest,
) -> dict[str, Any]:
    """Update an existing LLM provider.

    Requires admin authentication.
    """
    # TODO: Implement provider update
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Provider update not yet implemented",
    )


@router.delete("/llm/providers/{name}", dependencies=[Security(autogpt_libs.auth.requires_admin_user)], status_code=status.HTTP_204_NO_CONTENT)
async def delete_provider(
    name: str,
) -> None:
    """Delete an LLM provider.

    Requires admin authentication.
    """
    # TODO: Implement provider deletion
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Provider deletion not yet implemented",
    )
