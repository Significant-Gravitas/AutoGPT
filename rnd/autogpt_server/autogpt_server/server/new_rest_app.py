from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from autogpt_server.server.utils import get_user_id

from autogpt_server.server.routes import root_router, agents_router, blocks_router

app = FastAPI()

app.include_router(root_router)
app.include_router(agents_router)
app.include_router(blocks_router)

def handle_internal_http_error(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": str(exc)},
    )

app.add_exception_handler(500, handle_internal_http_error)


