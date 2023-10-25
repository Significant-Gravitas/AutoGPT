import uuid
from abc import ABC
from typing import Dict, List, Optional, Any
from .models.task import Task
from .models.step import Step
from .models.artifact import Artifact
from .legacy_models import Status, NotFoundResponse
from fastapi.responses import JSONResponse
from fastapi import Request
from datetime import datetime


# class Step(APIStep):
#    additional_input: Optional[Dict[str, str]] = None


# class Task(APITask):
#    steps: List[Step] = []


class NotFoundException(Exception):
    """
    Exception raised when a resource is not found.
    """

    def __init__(self, item_name: str, item_id: str):
        self.item_name = item_name
        self.item_id = item_id
        super().__init__(
            NotFoundResponse(message=f"{item_name} with {item_id} not found.")
        )


async def not_found_exception_handler(
    request: Request, exc: NotFoundException
) -> JSONResponse:
    return JSONResponse(
        content={"message": f"{exc.item_name} with {exc.item_id} not found."},
        status_code=404,
    )


class AbstractDB(ABC):
    async def create_task(
        self,
        input: Optional[str],
        additional_input: Any = None,
        artifacts: Optional[List[Artifact]] = None,
    ) -> Task:
        raise NotImplementedError

    async def create_step(
        self,
        task_id: str,
        name: Optional[str] = None,
        input: Optional[str] = None,
        is_last: bool = False,
        additional_input: Optional[Dict[str, str]] = None,
        artifacts: List[Artifact] = [],
    ) -> Step:
        raise NotImplementedError

    async def create_artifact(
        self,
        task_id: str,
        file_name: str,
        agent_created: bool = True,
        relative_path: Optional[str] = None,
        step_id: Optional[str] = None,
    ) -> Artifact:
        raise NotImplementedError

    async def get_task(self, task_id: str) -> Task:
        raise NotImplementedError

    async def get_step(self, task_id: str, step_id: str) -> Step:
        raise NotImplementedError

    async def get_artifact(self, task_id: str, artifact_id: str) -> Artifact:
        raise NotImplementedError

    async def list_tasks(self) -> List[Task]:
        raise NotImplementedError

    async def list_steps(self, task_id: str, status: Optional[str] = None) -> List[Step]:
        raise NotImplementedError

    async def list_artifacts(
        self,
        task_id: str,
    ) -> List[Artifact]:
        raise NotImplementedError


class InMemorySeparatedDB(AbstractDB):
    _tasks: Dict[str, Task] = {}
    _steps: Dict[str, Step] = {}
    _artifacts: Dict[str, Artifact] = {}

    async def create_task(
        self,
        input: Optional[str],
        additional_input: Any = None,
        artifacts: Optional[List[str]] = None,
    ) -> Task:
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            input=input,
            additional_input=additional_input,
            artifacts=[],
        )
        self._tasks[task_id] = task
        return task

    async def create_step(
        self,
        task_id: str,
        name: Optional[str] = None,
        input: Optional[str] = None,
        is_last=False,
        additional_input: Optional[Dict[str, Any]] = None,
        artifacts: List[str] = [],
    ) -> Step:
        step_id = str(uuid.uuid4())
        step = Step(
            task_id=task_id,
            step_id=step_id,
            name=name,
            input=input,
            status="created",
            is_last=is_last,
            additional_input=additional_input,
            artifacts=[],
        )
        self._steps[step_id] = step
        return step

    async def create_artifact(
        self,
        task_id: str,
        file_name: str,
        agent_created: bool = True,
        relative_path: Optional[str] = None,
        step_id: Optional[str] = None,
    ) -> Artifact:
        artifact_id = str(uuid.uuid4())
        artifact = Artifact(
            artifact_id=artifact_id,
            agent_created=agent_created,
            file_name=file_name,
            relative_path=relative_path,
            created_at=str(datetime.now()),
            modifed_at=str(datetime.now()),
        )
        self._artifacts[artifact_id] = artifact
        task = await self.get_task(task_id)
        task.artifacts.append(artifact)
        try:
            step = await self.get_step(task_id, step_id)
            step.artifacts.append(artifact)
        except NotFoundException:
            pass
        return artifact

    async def get_task(self, task_id: str) -> Task:
        task = self._tasks.get(task_id, None)
        if not task:
            raise NotFoundException("Task", task_id)
        return task

    async def get_step(self, task_id: str, step_id: str) -> Step:
        step = self._steps.get(step_id, None)
        if not step or step.task_id != task_id:
            raise NotFoundException("Step", step_id)
        return step

    async def get_artifact(self, task_id: str, artifact_id: str) -> Artifact:
        artifact = self._artifacts.get(artifact_id, None)
        if not artifact:
            raise NotFoundException("Artifact", artifact_id)
        return artifact

    async def list_tasks(self) -> List[Task]:
        return list(self._tasks.values())

    async def list_steps(self, task_id: str, status: Optional[str] = None) -> List[Step]:
        steps = [step for step in self._steps.values() if step.task_id == task_id]
        if status:
            steps = list(filter(lambda s: s.status == status, steps))
        return steps

    async def list_artifacts(self, task_id: str) -> List[Artifact]:
        task = await self.get_task(task_id)
        return task.artifacts
