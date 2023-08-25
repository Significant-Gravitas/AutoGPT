"""
This is an example implementation of the Agent Protocol DB for development Purposes
It uses SQLite as the database and file store backend.
IT IS NOT ADVISED TO USE THIS IN PRODUCTION!
"""

import math
from typing import Dict, List, Optional, Tuple

from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, joinedload, relationship, sessionmaker

from .forge_log import CustomLogger
from .schema import Artifact, Pagination, Status, Step, Task, TaskInput

LOG = CustomLogger(__name__)


class Base(DeclarativeBase):
    pass


class DataNotFoundError(Exception):
    pass


class TaskModel(Base):
    __tablename__ = "tasks"

    task_id = Column(Integer, primary_key=True, autoincrement=True)
    input = Column(String)
    additional_input = Dict

    artifacts = relationship("ArtifactModel", back_populates="task")


class StepModel(Base):
    __tablename__ = "steps"

    step_id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey("tasks.task_id"))
    name = Column(String)
    input = Column(String)
    status = Column(String)
    is_last = Column(Boolean, default=False)

    additional_properties = Column(String)
    artifacts = relationship("ArtifactModel", back_populates="step")


class ArtifactModel(Base):
    __tablename__ = "artifacts"

    artifact_id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey("tasks.task_id"))
    step_id = Column(Integer, ForeignKey("steps.step_id"))
    agent_created = Column(Boolean, default=False)
    file_name = Column(String)
    uri = Column(String)

    step = relationship("StepModel", back_populates="artifacts")
    task = relationship("TaskModel", back_populates="artifacts")


def convert_to_task(task_obj: TaskModel, debug_enabled: bool = False) -> Task:
    if debug_enabled:
        LOG.debug(f"Converting TaskModel to Task for task_id: {task_obj.task_id}")
    task_artifacts = [
        Artifact(
            artifact_id=artifact.artifact_id,
            file_name=artifact.file_name,
            agent_created=artifact.agent_created,
            uri=artifact.uri,
        )
        for artifact in task_obj.artifacts
        if artifact.task_id == task_obj.task_id
    ]
    return Task(
        task_id=task_obj.task_id,
        input=task_obj.input,
        additional_input=task_obj.additional_input,
        artifacts=task_artifacts,
    )


def convert_to_step(step_model: StepModel, debug_enabled: bool = False) -> Step:
    if debug_enabled:
        LOG.debug(f"Converting StepModel to Step for step_id: {step_model.step_id}")
    step_artifacts = [
        Artifact(
            artifact_id=artifact.artifact_id,
            file_name=artifact.file_name,
            agent_created=artifact.agent_created,
            uri=artifact.uri,
        )
        for artifact in step_model.artifacts
        if artifact.step_id == step_model.step_id
    ]
    status = Status.completed if step_model.status == "completed" else Status.created
    return Step(
        task_id=step_model.task_id,
        step_id=step_model.step_id,
        name=step_model.name,
        input=step_model.input,
        status=status,
        artifacts=step_artifacts,
        is_last=step_model.is_last == 1,
        additional_properties=step_model.additional_properties,
    )


# sqlite:///{database_name}
class AgentDB:
    def __init__(self, database_string, debug_enabled: bool = False) -> None:
        super().__init__()
        self.debug_enabled = debug_enabled
        if self.debug_enabled:
            LOG.debug(f"Initializing AgentDB with database_string: {database_string}")
        self.engine = create_engine(database_string)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    async def create_task(
        self, input: Optional[str], additional_input: Optional[TaskInput] = None
    ) -> Task:
        if self.debug_enabled:
            LOG.debug("Creating new task")
        with self.Session() as session:
            new_task = TaskModel(
                input=input,
                additional_input=additional_input.__root__
                if additional_input
                else None,
            )
            session.add(new_task)
            session.commit()
            session.refresh(new_task)
            if self.debug_enabled:
                LOG.debug(f"Created new task with task_id: {new_task.task_id}")
            return convert_to_task(new_task, self.debug_enabled)

    async def create_step(
        self,
        task_id: str,
        input: str,
        is_last: bool = False,
        additional_properties: Optional[Dict[str, str]] = None,
    ) -> Step:
        if self.debug_enabled:
            LOG.debug(f"Creating new step for task_id: {task_id}")
        try:
            session = self.Session()
            new_step = StepModel(
                task_id=task_id,
                name=input,
                input=input,
                status="created",
                is_last=is_last,
                additional_properties=additional_properties,
            )
            session.add(new_step)
            session.commit()
            session.refresh(new_step)
            if self.debug_enabled:
                LOG.debug(f"Created new step with step_id: {new_step.step_id}")
        except Exception as e:
            LOG.error(f"Error while creating step: {e}")
        return convert_to_step(new_step, self.debug_enabled)

    async def create_artifact(
        self,
        task_id: str,
        file_name: str,
        uri: str,
        agent_created: bool = False,
        step_id: str | None = None,
    ) -> Artifact:
        if self.debug_enabled:
            LOG.debug(f"Creating new artifact for task_id: {task_id}")
        session = self.Session()

        if existing_artifact := session.query(ArtifactModel).filter_by(uri=uri).first():
            session.close()
            if self.debug_enabled:
                LOG.debug(f"Artifact already exists with uri: {uri}")
            return Artifact(
                artifact_id=str(existing_artifact.artifact_id),
                file_name=existing_artifact.file_name,
                agent_created=existing_artifact.agent_created,
                uri=existing_artifact.uri,
            )

        new_artifact = ArtifactModel(
            task_id=task_id,
            step_id=step_id,
            agent_created=agent_created,
            file_name=file_name,
            uri=uri,
        )
        session.add(new_artifact)
        session.commit()
        session.refresh(new_artifact)
        if self.debug_enabled:
            LOG.debug(
                f"Created new artifact with artifact_id: {new_artifact.artifact_id}"
            )
        return Artifact(
            artifact_id=str(new_artifact.artifact_id),
            file_name=new_artifact.file_name,
            agent_created=new_artifact.agent_created,
            uri=new_artifact.uri,
        )

    async def get_task(self, task_id: int) -> Task:
        """Get a task by its id"""
        if self.debug_enabled:
            LOG.debug(f"Getting task with task_id: {task_id}")
        session = self.Session()
        if task_obj := (
            session.query(TaskModel)
            .options(joinedload(TaskModel.artifacts))
            .filter_by(task_id=task_id)
            .first()
        ):
            return convert_to_task(task_obj, self.debug_enabled)
        else:
            LOG.error(f"Task not found with task_id: {task_id}")
            raise DataNotFoundError("Task not found")

    async def get_step(self, task_id: int, step_id: int) -> Step:
        if self.debug_enabled:
            LOG.debug(f"Getting step with task_id: {task_id} and step_id: {step_id}")
        session = self.Session()
        if step := (
            session.query(StepModel)
            .options(joinedload(StepModel.artifacts))
            .filter(StepModel.step_id == step_id)
            .first()
        ):
            return convert_to_step(step, self.debug_enabled)

        else:
            LOG.error(f"Step not found with task_id: {task_id} and step_id: {step_id}")
            raise DataNotFoundError("Step not found")

    async def update_step(
        self,
        task_id: str,
        step_id: str,
        status: str,
        additional_properties: Optional[Dict[str, str]] = None,
    ) -> Step:
        if self.debug_enabled:
            LOG.debug(f"Updating step with task_id: {task_id} and step_id: {step_id}")
        session = self.Session()
        if (
            step := session.query(StepModel)
            .filter_by(task_id=task_id, step_id=step_id)
            .first()
        ):
            step.status = status
            step.additional_properties = additional_properties
            session.commit()
            return await self.get_step(task_id, step_id)
        else:
            LOG.error(
                f"Step not found for update with task_id: {task_id} and step_id: {step_id}"
            )
            raise DataNotFoundError("Step not found")

    async def get_artifact(self, task_id: str, artifact_id: str) -> Artifact:
        if self.debug_enabled:
            LOG.debug(
                f"Getting artifact with task_id: {task_id} and artifact_id: {artifact_id}"
            )
        session = self.Session()
        if artifact_model := (
            session.query(ArtifactModel)
            .filter_by(task_id=int(task_id), artifact_id=int(artifact_id))
            .first()
        ):
            return Artifact(
                artifact_id=artifact_model.artifact_id,  # Casting to string
                file_name=artifact_model.file_name,
                agent_created=artifact_model.agent_created,
                uri=artifact_model.uri,
            )
        else:
            LOG.error(
                f"Artifact not found with task_id: {task_id} and artifact_id: {artifact_id}"
            )
            raise DataNotFoundError("Artifact not found")

    async def list_tasks(
        self, page: int = 1, per_page: int = 10
    ) -> Tuple[List[Task], Pagination]:
        if self.debug_enabled:
            LOG.debug("Listing tasks")
        session = self.Session()
        tasks = (
            session.query(TaskModel).offset((page - 1) * per_page).limit(per_page).all()
        )
        total = session.query(TaskModel).count()
        pages = math.ceil(total / per_page)
        pagination = Pagination(
            total_items=total, total_pages=pages, current_page=page, page_size=per_page
        )
        return [convert_to_task(task, self.debug_enabled) for task in tasks], pagination

    async def list_steps(
        self, task_id: str, page: int = 1, per_page: int = 10
    ) -> Tuple[List[Step], Pagination]:
        if self.debug_enabled:
            LOG.debug(f"Listing steps for task_id: {task_id}")
        session = self.Session()
        steps = (
            session.query(StepModel)
            .filter_by(task_id=task_id)
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )
        total = session.query(StepModel).filter_by(task_id=task_id).count()
        pages = math.ceil(total / per_page)
        pagination = Pagination(
            total_items=total, total_pages=pages, current_page=page, page_size=per_page
        )
        return [convert_to_step(step, self.debug_enabled) for step in steps], pagination

    async def list_artifacts(
        self, task_id: str, page: int = 1, per_page: int = 10
    ) -> Tuple[List[Artifact], Pagination]:
        if self.debug_enabled:
            LOG.debug(f"Listing artifacts for task_id: {task_id}")
        with self.Session() as session:
            artifacts = (
                session.query(ArtifactModel)
                .filter_by(task_id=task_id)
                .offset((page - 1) * per_page)
                .limit(per_page)
                .all()
            )
            total = session.query(ArtifactModel).filter_by(task_id=task_id).count()
            pages = math.ceil(total / per_page)
            pagination = Pagination(
                total_items=total,
                total_pages=pages,
                current_page=page,
                page_size=per_page,
            )
            return [
                Artifact(
                    artifact_id=str(artifact.artifact_id),
                    file_name=artifact.file_name,
                    agent_created=artifact.agent_created,
                    uri=artifact.uri,
                )
                for artifact in artifacts
            ], pagination
