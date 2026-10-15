from autogpt_libs.auth import get_user_id, requires_user
from fastapi import APIRouter, HTTPException, Path, Response, Security
from fastapi.concurrency import run_in_threadpool

from backend.api.features.experts.package_export import (
    build_template_package,
    zip_from_expert_package,
)
from backend.api.features.experts.package_model import (
    ExpertPackageError,
    package_filename,
)
from backend.util.exceptions import NotFoundError

# Signed-in only: each download builds every workflow's graph, which is not
# work to hand anonymous traffic.
router = APIRouter(dependencies=[Security(requires_user)])


@router.get(
    "/templates/{template_id}/package",
    summary="Download an expert template as a .expert.zip file",
    operation_id="download_expert_template_package",
    response_class=Response,
    responses={
        200: {
            "content": {
                "application/zip": {"schema": {"type": "string", "format": "binary"}}
            },
            "description": "The template's manifest, skills and workflows as a zip",
        },
        404: {"description": "Expert template not found"},
        413: {"description": "The package is over the size limit"},
    },
)
async def download_expert_template_package(
    user_id: str = Security(get_user_id),
    template_id: str = Path(..., description="ID of the marketplace template"),
) -> Response:
    """Return the template as ``<name>.expert.zip``: ``expert.json``, each
    bundled skill under ``skills/<slug>/`` and each workflow under
    ``workflows/``. Never memory, conversations or files; a template has
    none, and the format has nowhere to put them."""
    try:
        package = await build_template_package(template_id, user_id)
        content = await run_in_threadpool(zip_from_expert_package, package)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail="Expert not found") from exc
    except ExpertPackageError as exc:
        raise HTTPException(
            status_code=413 if exc.over_limit else 400, detail=str(exc)
        ) from exc
    filename = package_filename(package.manifest.name)
    return Response(
        content=content,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
