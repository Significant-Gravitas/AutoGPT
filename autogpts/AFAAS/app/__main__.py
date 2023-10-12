import os

import uvicorn
from dotenv import load_dotenv

from routes import agent_router, app_router, artifact_router
import app.sdk.forge_log
LOG = app.sdk.forge_log.ForgeLogger(__name__)


logo = """\n\n
    A      FFFFF   A     A       SSSSSS 
   A A     F      A A   A A     S     
  A   A    FFFF  A   A A   A    SSSSS  
 AAAAAAA   F    AAAAAAAAAAAAA        S 
A       A  F   A     AA      ASSSSSSSS 
\n\n"""






if __name__ == "__main__":
    print(logo)
    port = os.getenv("PORT", 8000)
    LOG.info(f"Agent server starting on http://localhost:{port}")
    load_dotenv()
    import os
    import logging
    import pathlib
    from io import BytesIO
    from uuid import uuid4

    import uvicorn
    from fastapi import APIRouter, FastAPI, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import RedirectResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles


    api = FastAPI(
        title="AutoGPT Forge",
        description="Modified version of The Agent Protocol.",
        version="v0.4",
    )
    api.include_router(agent_router, prefix="/agent")
    api.include_router(app_router, prefix="/app")
    api.include_router(artifact_router, prefix="/artifact")

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

    script_dir = os.path.dirname(os.path.realpath(__file__))
    frontend_path = pathlib.Path(
        os.path.join(script_dir, "../../../../frontend/build/web")
    ).resolve()

    if os.path.exists(frontend_path):
        api.mount("/app", StaticFiles(directory=frontend_path), name="app")

        @api.get("/", include_in_schema=False)
        async def root():
            return RedirectResponse(url="/app/index.html", status_code=307)

    else:
        logging.Logger.warning(
            f"Frontend not found. {frontend_path} does not exist. The frontend will not be served"
        )

    uvicorn.run(
        "main:app", host="localhost", port=port, log_level="error", reload=True
    )
