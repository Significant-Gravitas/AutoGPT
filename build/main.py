from multiprocessing import freeze_support
from fastapi.responses import Response
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def index() -> Response:
    return Response(status_code=204)
