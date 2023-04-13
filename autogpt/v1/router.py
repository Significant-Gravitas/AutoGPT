from fastapi.routing import APIRouter

from autogpt.v1.endpoints import docs, health, agents

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(docs.router)
api_router.include_router(agents.router)
