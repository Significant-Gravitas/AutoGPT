from fastapi.routing import APIRouter

from autogpt.api.v1.endpoints import docs, health, tasks, agents, interactions

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(docs.router)
api_router.include_router(tasks.router)
api_router.include_router(agents.router)
api_router.include_router(interactions.router)
