from fastapi import FastAPI

from autogpt.core.server.routers import agent

app = FastAPI()

app.include_router(agent.router, prefix="/agents", tags=["Agent Management"])


@app.get("/")
def read_root():
    return {"message": "Welcome to your SaaS API!"}
