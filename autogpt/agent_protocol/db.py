import uuid
from abc import ABC
from typing import Any, Dict, List, Optional

from .models import Artifact, Status
from .models import Step as APIStep
from .models import Task as APITask


class Step(APIStep):
    additional_properties: Optional[Dict[str, str]] = None


class Task(APITask):
    steps: Optional[List[Step]] = []


class NotFoundException(Exception):
    """
    Exception raised when a resource is not found.
    """

    def __init__(self, item_name: str, item_id: str):
        self.item_name = item_name
        self.item_id = item_id
        super().__init__(f"{item_name} with {item_id} not found.")


class TaskDB(ABC):
    async def create_task(
        self, input: Optional[str], additional_input: Any = None
    ) -> Task:
        raise NotImplementedError

    async def create_step(
        self,
        task_id: str,
        name: Optional[str] = None,
        input: Optional[str] = None,
        is_last: bool = False,
        additional_properties: Optional[Dict[str, str]] = None,
    ) -> Step:
        raise NotImplementedError

    async def create_artifact(
        self,
        task_id: str,
        artifact: Artifact,
        step_id: str | None = None,
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

    async def list_steps(
        self, task_id: str, status: Optional[Status] = None
    ) -> List[Step]:
        raise NotImplementedError

    async def update_step(
        self,
        task_id: str,
        step_id: str,
        status: str,
        additional_properties: Optional[Dict[str, str]] = None,
    ) -> Step:
        raise NotImplementedError


class InMemoryTaskDB(TaskDB):
    _tasks: Dict[str, Task] = {}

    async def create_task(
        self,
        input: Optional[str],
        additional_input: Any = None,
    ) -> Task:
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            input=input,
            steps=[],
            artifacts=[],
            additional_input=additional_input,
        )
        self._tasks[task_id] = task
        return task

    async def create_step(
        self,
        task_id: str,
        name: Optional[str] = None,
        input: Optional[str] = None,
        is_last=False,
        additional_properties: Optional[Dict[str, Any]] = None,
    ) -> Step:
        step_id = str(uuid.uuid4())
        step = Step(
            task_id=task_id,
            step_id=step_id,
            name=name,
            input=input,
            status=Status.created,
            is_last=is_last,
            additional_properties=additional_properties,
        )
        task = await self.get_task(task_id)
        task.steps.append(step)
        return step

    async def get_task(self, task_id: str) -> Task:
        task = self._tasks.get(task_id, None)
        if not task:
            raise NotFoundException("Task", task_id)
        return task

    async def get_step(self, task_id: str, step_id: str) -> Step:
        task = await self.get_task(task_id)
        step = next(filter(lambda s: s.task_id == task_id, task.steps), None)
        if not step:
            raise NotFoundException("Step", step_id)
        return step

    async def get_artifact(self, task_id: str, artifact_id: str) -> Artifact:
        task = await self.get_task(task_id)
        artifact = next(
            filter(lambda a: a.artifact_id == artifact_id, task.artifacts), None
        )
        if not artifact:
            raise NotFoundException("Artifact", artifact_id)
        return artifact

    async def create_artifact(
        self,
        task_id: str,
        file_name: str,
        relative_path: Optional[str] = None,
        step_id: Optional[str] = None,
    ) -> Artifact:
        artifact_id = str(uuid.uuid4())
        artifact = Artifact(
            artifact_id=artifact_id, file_name=file_name, relative_path=relative_path
        )
        task = await self.get_task(task_id)
        task.artifacts.append(artifact)

        if step_id:
            step = await self.get_step(task_id, step_id)
            step.artifacts.append(artifact)

        return artifact

    async def list_tasks(self) -> List[Task]:
        return [task for task in self._tasks.values()]

    async def list_steps(
        self, task_id: str, status: Optional[Status] = None
    ) -> List[Step]:
        task = await self.get_task(task_id)
        steps = task.steps
        if status:
            steps = list(filter(lambda s: s.status == status, steps))
        return steps
