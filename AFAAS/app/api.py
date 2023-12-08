import logging
import os
import pathlib
from io import BytesIO
from uuid import uuid4

import app.sdk.forge_log
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from routes import (AgentMiddleware, UserIDMiddleware, afaas_agent_router,
                    afaas_artifact_router, afaas_user_router, agent_router,
                    app_router, artifact_router, user_router)

LOG = app.sdk.forge_log.ForgeLogger(__name__)

port = os.getenv("PORT", 8000)
load_dotenv()
LOG.info(f"Agent server starting on http://localhost:{port}")

api = FastAPI(
    title="AutoGPT Forge",
    description="Modified version of The Agent Protocol.",
    version="v0.4",
)
# Add CORS middleware
origins = [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    # Add any other origins you want to whitelist
]

api.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


api.add_middleware(UserIDMiddleware)
api.include_router(app_router)
api.include_router(user_router, prefix="/ap/v1")
api.include_router(afaas_user_router, prefix="/afaas/v1")


# api.add_middleware(AgentMiddleware)
api.include_router(agent_router, prefix="/ap/v1")
api.include_router(afaas_agent_router, prefix="/afaas/v1")

api.include_router(artifact_router, prefix="/ap/v1")
api.include_router(afaas_artifact_router, prefix="/afaas/v1")


script_dir = os.path.dirname(os.path.realpath(__file__))
frontend_path = pathlib.Path(
    os.path.join(script_dir, "../../../frontend/build/web")
).resolve()

if os.path.exists(frontend_path):
    api.mount("/app", StaticFiles(directory=frontend_path), name="app")

    @api.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/app/index.html", status_code=307)

else:
    LOG.warning(
        f"Frontend not found. {frontend_path} does not exist. The frontend will not be served"
    )
