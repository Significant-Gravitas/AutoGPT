import autogpt_libs.auth
import fastapi

from backend.server.v2.llm import db as llm_db
from backend.server.v2.llm import model as llm_model

router = fastapi.APIRouter(
    prefix="/llm",
    tags=["llm"],
    dependencies=[fastapi.Security(autogpt_libs.auth.requires_user)],
)


@router.get("/models", response_model=llm_model.LlmModelsResponse)
async def list_models():
    models = await llm_db.list_models()
    return llm_model.LlmModelsResponse(models=models)


@router.get("/providers", response_model=llm_model.LlmProvidersResponse)
async def list_providers():
    providers = await llm_db.list_providers(include_models=True)
    return llm_model.LlmProvidersResponse(providers=providers)

