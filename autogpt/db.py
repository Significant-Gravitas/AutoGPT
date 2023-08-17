"""
This is an example implementation of the Agent Protocol DB for development Purposes
It uses SQLite as the database and file store backend.
IT IS NOT ADVISED TO USE THIS IN PRODUCTION!
"""

from typing import Dict, List, Optional

from agent_protocol import Artifact, Step, Task, TaskDB
from agent_protocol.models import Status, TaskInput
from sqlalchemy import (
    Boolean,
    Column,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class DataNotFoundError(Exception):
    pass


class TaskModel(Base):
    __tablename__ = "tasks"

    task_id = Column(Integer, primary_key=True, autoincrement=True)
    input = Column(String)
    additional_input = Column(String)

    steps = relationship("StepModel", back_populates="task")
    artifacts = relationship("ArtifactModel", back_populates="task")


class StepModel(Base):
    __tablename__ = "steps"

    step_id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey("tasks.task_id"))
    name = Column(String)
    status = Column(String)
    is_last = Column(Boolean, default=False)
    additional_properties = Column(String)

    task = relationship("TaskModel", back_populates="steps")


class ArtifactModel(Base):
    __tablename__ = "artifacts"

    artifact_id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey("tasks.task_id"))
    step_id = Column(Integer, ForeignKey("steps.step_id"))
    file_name = Column(String)
    relative_path = Column(String)
    file_data = Column(LargeBinary)

    task = relationship("TaskModel", back_populates="artifacts")


# sqlite:///{database_name}
class AgentDB(TaskDB):
    def __init__(self, database_string) -> None:
        super().__init__()
        self.engine = create_engine(database_string)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        print("Databases Created")

    async def create_task(
        self,
        input: Optional[str],
        additional_input: Optional[TaskInput] = None,
        artifacts: List[Artifact] = None,
        steps: List[Step] = None,
    ) -> Task:
        session = self.Session()
        new_task = TaskModel(
            input=input,
            additional_input=additional_input.json() if additional_input else None,
        )
        session.add(new_task)
        session.commit()
        session.refresh(new_task)
        return await self.get_task(new_task.task_id)

    async def create_step(
        self,
        task_id: str,
        name: Optional[str] = None,
        is_last: bool = False,
        additional_properties: Optional[Dict[str, str]] = None,
    ) -> Step:
        session = self.Session()
        new_step = StepModel(
            task_id=task_id,
            name=name,
            status="created",
            is_last=is_last,
            additional_properties=additional_properties,
        )
        session.add(new_step)
        session.commit()
        session.refresh(new_step)
        return await self.get_step(task_id, new_step.step_id)

    async def create_artifact(
        self,
        task_id: str,
        file_name: str,
        relative_path: Optional[str] = None,
        step_id: Optional[str] = None,
        file_data: bytes | None = None,
    ) -> Artifact:
        session = self.Session()
        new_artifact = ArtifactModel(
            task_id=task_id,
            step_id=step_id,
            file_name=file_name,
            relative_path=relative_path,
            file_data=file_data,
        )
        session.add(new_artifact)
        session.commit()
        session.refresh(new_artifact)
        return await self.get_artifact(task_id, new_artifact.artifact_id)

    async def get_task(self, task_id: int) -> Task:
        """Get a task by its id"""
        session = self.Session()
        task_obj = session.query(TaskModel).filter_by(task_id=task_id).first()
        if task_obj:
            task = Task(
                task_id=task_obj.task_id,
                input=task_obj.input,
                additional_input=task_obj.additional_input,
                steps=[],
            )
            steps_obj = session.query(StepModel).filter_by(task_id=task_id).all()
            if steps_obj:
                for step in steps_obj:
                    status = (
                        Status.created if step.status == "created" else Status.completed
                    )
                    task.steps.append(
                        Step(
                            task_id=step.task_id,
                            step_id=step.step_id,
                            name=step.name,
                            status=status,
                            is_last=step.is_last == 1,
                            additional_properties=step.additional_properties,
                        )
                    )
            return task
        else:
            raise DataNotFoundError("Task not found")

    async def get_step(self, task_id: int, step_id: int) -> Step:
        session = self.Session()
        if (
            step := session.query(StepModel)
            .filter_by(task_id=task_id, step_id=step_id)
            .first()
        ):
            status = Status.completed if step.status == "completed" else Status.created
            return Step(
                task_id=task_id,
                step_id=step_id,
                name=step.name,
                status=status,
                is_last=step.is_last == 1,
                additional_properties=step.additional_properties,
            )
        else:
            raise DataNotFoundError("Step not found")

    async def update_step(
        self,
        task_id: str,
        step_id: str,
        status: str,
        additional_properties: Optional[Dict[str, str]] = None,
    ) -> Step:
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
            raise DataNotFoundError("Step not found")

    async def get_artifact(self, task_id: str, artifact_id: str) -> Artifact:
        session = self.Session()
        if (
            artifact := session.query(ArtifactModel)
            .filter_by(task_id=task_id, artifact_id=artifact_id)
            .first()
        ):
            return Artifact(
                artifact_id=artifact.artifact_id,
                file_name=artifact.file_name,
                relative_path=artifact.relative_path,
            )
        else:
            raise DataNotFoundError("Artifact not found")

    async def get_artifact_file(self, task_id: str, artifact_id: str) -> bytes:
        session = self.Session()
        if (
            artifact := session.query(ArtifactModel.file_data)
            .filter_by(task_id=task_id, artifact_id=artifact_id)
            .first()
        ):
            return artifact.file_data
        else:
            raise DataNotFoundError("Artifact not found")

    async def list_tasks(self) -> List[Task]:
        session = self.Session()
        tasks = session.query(TaskModel).all()
        return [
            Task(
                task_id=task.task_id,
                input=task.input,
                additional_input=task.additional_input,
            )
            for task in tasks
        ]

    async def list_steps(self, task_id: str) -> List[Step]:
        session = self.Session()
        steps = session.query(StepModel).filter_by(task_id=task_id).all()
        return [
            Step(
                task_id=task_id,
                step_id=step.step_id,
                name=step.name,
                status=step.status,
            )
            for step in steps
        ]
