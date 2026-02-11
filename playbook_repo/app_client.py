from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
import base64
import sys
import os

# Adjust path to find the playbook package if running directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from playbook import config
from playbook.brain_client import BrainClient
from playbook.trainer import Trainer
from playbook.executor import Executor
from playbook.utils import get_logger, generate_uuid

logger = get_logger("app_client")

# Global instances
brain_client = None
trainer = None
executor = None

class TrainRequest(BaseModel):
    session_id: str
    mapping: list

class FinishRequest(BaseModel):
    session_id: str

class ExecuteRequest(BaseModel):
    session_id: str
    files: list

@asynccontextmanager
async def lifespan(app: FastAPI):
    global brain_client, trainer, executor
    logger.info("Starting up Playbook Client...")
    logger.info(config.get_masked_info())

    brain_client = BrainClient()
    logger.info("Performing startup health probe...")
    try:
        brain_client.check_health()
    except RuntimeError as e:
        logger.critical(f"Startup probe failed: {e}")
        # Stop process
        sys.exit(1)

    trainer = Trainer(brain_client)
    executor = Executor(brain_client)
    logger.info("System ready.")
    yield
    logger.info("Shutting down Playbook Client...")

app = FastAPI(title="Playbook Client", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), session_id: str = Form(None)):
    if not session_id or session_id == "null":
        session_id = generate_uuid()

    logger.info(f"Uploading file {file.filename} for session {session_id}")
    try:
        content = await file.read()
        content_b64 = base64.b64encode(content).decode('utf-8')
        result = await brain_client.extract(session_id, file.filename, content_b64)
        return result
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train")
async def train_endpoint(req: TrainRequest):
    try:
        return await trainer.submit_mapping(req.session_id, req.mapping)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/finish_training")
async def finish_training_endpoint(req: FinishRequest):
    try:
        return await trainer.finish(req.session_id)
    except Exception as e:
        logger.error(f"Finish failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/execute")
async def execute_endpoint(req: ExecuteRequest):
    try:
        return await executor.execute_files(req.session_id, req.files)
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/opportunities")
async def opportunities_endpoint():
    try:
        return await brain_client.get_opportunities()
    except Exception as e:
        logger.error(f"Opportunities failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Static Files
# Only if directory exists
if os.path.isdir("ui"):
    app.mount("/", StaticFiles(directory="ui", html=True), name="ui")
else:
    logger.warning("UI directory not found. Static files not served.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
