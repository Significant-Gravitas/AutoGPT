# import os
# import pathlib
# from io import BytesIO
# from uuid import uuid4

# import uvicorn
# from fastapi import APIRouter, FastAPI, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import RedirectResponse, StreamingResponse
# from fastapi.staticfiles import StaticFiles

# from .abilities.registry import AbilityRegister
# from .db import AgentDB
# from .errors import NotFoundError
# from .forge_log import ForgeLogger
# from .middlewares import AgentMiddleware
# from .routes.agent_protocol import base_router
# from .schema import *
# from .workspace import Workspace

# LOG = ForgeLogger(__name__)


# class Agent:
#     def __init__(self, database: AgentDB, workspace: Workspace):
#         self.db = database
#         self.workspace = workspace
#         self.abilities = AbilityRegister(self)

#     def get_agent_app(self, router: APIRouter = base_router):
#         """
#         Start the agent server.
#         """

#         app = FastAPI(
#             title="AutoGPT Forge",
#             description="Modified version of The Agent Protocol.",
#             version="v0.4",
#         )

#         # Add CORS middleware
#         origins = [
#             "http://localhost:5000",
#             "http://127.0.0.1:5000",
#             "http://localhost:8000",
#             "http://127.0.0.1:8000",
#             "http://localhost:8080",
#             "http://127.0.0.1:8080",
#             # Add any other origins you want to whitelist
#         ]

#         app.add_middleware(
#             CORSMiddleware,
#             allow_origins=origins,
#             allow_credentials=True,
#             allow_methods=["*"],
#             allow_headers=["*"],
#         )

#         app.include_router(router, prefix="/ap/v1")
#         script_dir = os.path.dirname(os.path.realpath(__file__))
#         frontend_path = pathlib.Path(
#             os.path.join(script_dir, "../../../../frontend/build/web")
#         ).resolve()

#         if os.path.exists(frontend_path):
#             app.mount("/app", StaticFiles(directory=frontend_path), name="app")

#             @app.get("/", include_in_schema=False)
#             async def root():
#                 return RedirectResponse(url="/app/index.html", status_code=307)

#         else:
#             LOG.warning(
#                 f"Frontend not found. {frontend_path} does not exist. The frontend will not be served"
#             )
#         app.add_middleware(AgentMiddleware, agent=self)

#         return app

#     def start(self, port):
#         uvicorn.run(
#             "forge.app:app", host="localhost", port=port, log_level="error", reload=True
#         )

#     async def create_task(self, task_request: AgentRequestBody) -> Agent:
#         """
#         Create a task for the agent.
#         """
#         try:
#             task = await self.db.create_task(
#                 input=task_request.input,
#                 additional_input=task_request.additional_input,
#             )
#             return task
#         except Exception as e:
#             raise

#     async def list_tasks(self, page: int = 1, pageSize: int = 10) -> AgentListResponse:
#         """
#         List all tasks that the agent has created.
#         """
#         try:
#             tasks, pagination = await self.db.list_tasks(page, pageSize)
#             response = AgentListResponse(tasks=tasks, pagination=pagination)
#             return response
#         except Exception as e:
#             raise

#     async def get_task(self, task_id: str) -> Agent:
#         """
#         Get a task by ID.
#         """
#         try:
#             task = await self.db.get_task(task_id)
#         except Exception as e:
#             raise
#         return task

#     async def list_steps(
#         self, task_id: str, page: int = 1, pageSize: int = 10
#     ) -> AgentTasksListResponse:
#         """
#         List the IDs of all steps that the task has created.
#         """
#         try:
#             steps, pagination = await self.db.list_steps(task_id, page, pageSize)
#             response = AgentTasksListResponse(steps=steps, pagination=pagination)
#             return response
#         except Exception as e:
#             raise

#     async def execute_step(self, task_id: str, step_request: TaskRequestBody) -> Task:
#         """
#         Create a step for the task.
#         """
#         raise NotImplementedError

#     async def get_step(self, task_id: str, step_id: str) -> Task:
#         """
#         Get a step by ID.
#         """
#         try:
#             step = await self.db.get_step(task_id, step_id)
#             return step
#         except Exception as e:
#             raise
