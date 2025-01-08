import fastapi

import backend.server.v2.library.routes.agents
import backend.server.v2.library.routes.presets

router = fastapi.APIRouter()

router.include_router(backend.server.v2.library.routes.presets.router)
router.include_router(backend.server.v2.library.routes.agents.router)
