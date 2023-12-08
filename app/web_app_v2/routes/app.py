import json
from typing import Optional

from fastapi import APIRouter, Response

app_router = APIRouter()


@app_router.get("/", tags=["root"])
async def root():
    """
    Root endpoint that returns a welcome message.
    """
    return Response(content="Welcome to the AFAAS Demo")


@app_router.get("/heartbeat", tags=["server"])
async def check_server_status():
    """
    Check if the server is running.
    """
    return Response(content="Server is running.", status_code=200)
