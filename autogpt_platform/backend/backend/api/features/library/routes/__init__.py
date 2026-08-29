import fastapi

from .agents import router as agents_router
from .folders import router as folders_router
from .presets import router as presets_router

router = fastapi.APIRouter()

router.include_router(presets_router)
router.include_router(folders_router)
router.include_router(agents_router)
