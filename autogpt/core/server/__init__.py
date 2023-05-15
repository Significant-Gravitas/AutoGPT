from fastapi import FastAPI

from autogpt.core.server.routers import admin, agent, auth, user

app = FastAPI()

app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(user.router, prefix="/users", tags=["User Management"])
app.include_router(admin.router, prefix="/admin", tags=["Admin"])
app.include_router(agent.router, prefix="/agents", tags=["Agent Management"])


@app.get("/")
def read_root():
    return {"message": "Welcome to your SaaS API!"}
