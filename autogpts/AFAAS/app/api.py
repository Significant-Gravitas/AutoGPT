import os
import logging
import pathlib
from io import BytesIO
from uuid import uuid4
from dotenv import load_dotenv

from fastapi import APIRouter, FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from routes import (app_router, 
                    UserIDMiddleware, user_router, afaas_user_router,
                    AgentMiddleware, agent_router, afaas_agent_router,
                    artifact_router, afaas_artifact_router,             
)

import app.sdk.forge_log
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

# user_app = FastAPI()
# user_app.add_middleware(UserIDMiddleware)
# user_app.include_router(user_router, prefix="/ap/v1")
# api.mount("/", user_app)

# agent_app = FastAPI()
# agent_app.add_middleware(AgentMiddleware)
# agent_app.include_router(agent_router, prefix="/ap/v1")
# agent_app.include_router(app_router, prefix="/ap/v1")
# agent_app.include_router(artifact_router, prefix="/ap/v1")
# api.mount("/", user_app)


api.include_router(app_router, prefix="/ap/v1")

api.add_middleware(UserIDMiddleware)
api.include_router(user_router, prefix="/ap/v1")

# api.add_middleware(AgentMiddleware)
api.include_router(agent_router, prefix="/ap/v1")
api.include_router(artifact_router, prefix="/ap/v1")


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
    LOG.warning(f"Frontend not found. {frontend_path} does not exist. The frontend will not be served"
    )