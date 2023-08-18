"""
This is an example implementation of the Agent Protocol DB for development Purposes
It uses SQLite as the database and file store backend.
IT IS NOT ADVISED TO USE THIS IN PRODUCTION!
"""

from typing import Dict, List, Optional

from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, joinedload, relationship, sessionmaker

from autogpt.agent_protocol import Artifact, Step, Task, TaskDB
from autogpt.agent_protocol.models import Status, TaskInput


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
    input = Column(String)
    status = Column(String)
    is_last = Column(Boolean, default=False)
    additional_properties = Column(String)

    task = relationship("TaskModel", back_populates="steps")


class ArtifactModel(Base):
    __tablename__ = "artifacts"

    artifact_id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey("tasks.task_id"))
    step_id = Column(Integer, ForeignKey("steps.step_id"))
    agent_created = Column(Boolean, default=False)
    file_name = Column(String)
    uri = Column(String)

    task = relationship("TaskModel", back_populates="artifacts")


def convert_to_task(task_obj: TaskModel) -> Task:
    steps_list = []
    for step in task_obj.steps:
        status = Status.completed if step.status == "completed" else Status.created
        steps_list.append(
            Step(
                task_id=step.task_id,
                step_id=step.step_id,
                name=step.name,
                status=status,
                is_last=step.is_last == 1,
                additional_properties=step.additional_properties,
            )
        )
    return Task(
        task_id=task_obj.task_id,
        input=task_obj.input,
        additional_input=task_obj.additional_input,
        artifacts=[],
        steps=steps_list,
    )


def convert_to_step(step_model: StepModel) -> Step:
    print(step_model)
    step_artifacts = [
        Artifact(
            artifact_id=artifact.artifact_id,
            file_name=artifact.file_name,
            agent_created=artifact.agent_created,
            uri=artifact.uri,
        )
        for artifact in step_model.task.artifacts
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
class AgentDB(TaskDB):
    def __init__(self, database_string) -> None:
        super().__init__()
        self.engine = create_engine(database_string)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        print("Databases Created")

    async def create_task(
        self, input: Optional[str], additional_input: Optional[TaskInput] = None
    ) -> Task:
        session = self.Session()
        new_task = TaskModel(
            input=input,
            additional_input=additional_input.__root__ if additional_input else None,
        )
        session.add(new_task)
        session.commit()
        session.refresh(new_task)
        return convert_to_task(new_task)

    async def create_step(
        self,
        task_id: str,
        name: Optional[str] = None,
        input: Optional[str] = None,
        is_last: bool = False,
        additional_properties: Optional[Dict[str, str]] = None,
    ) -> Step:
        session = self.Session()
        new_step = StepModel(
            task_id=task_id,
            name=name,
            input=input,
            status="created",
            is_last=is_last,
            additional_properties=additional_properties,
        )
        session.add(new_step)
        session.commit()
        session.refresh(new_step)
        return convert_to_step(new_step)

    async def create_artifact(
        self,
        task_id: str,
        file_name: str,
        uri: str,
        agent_created: bool = False,
        step_id: str | None = None,
    ) -> Artifact:
        session = self.Session()

        if existing_artifact := session.query(ArtifactModel).filter_by(uri=uri).first():
            session.close()
            return existing_artifact

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
        return await self.get_artifact(task_id, new_artifact.artifact_id)

    async def get_task(self, task_id: int) -> Task:
        """Get a task by its id"""
        session = self.Session()
        task_obj = (
            session.query(TaskModel)
            .options(joinedload(TaskModel.steps))
            .filter_by(task_id=task_id)
            .first()
        )
        if task_obj:
            return convert_to_task(task_obj)
        else:
            raise DataNotFoundError("Task not found")

    async def get_step(self, task_id: int, step_id: int) -> Step:
        session = self.Session()
        if step := (
            session.query(StepModel)
            .options(joinedload(StepModel.task).joinedload(TaskModel.artifacts))
            .filter(StepModel.step_id == step_id)
            .first()
        ):
            return convert_to_step(step)

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
        if artifact_model := (
            session.query(ArtifactModel)
            .filter_by(task_id=task_id, artifact_id=artifact_id)
            .first()
        ):
            return Artifact(
                artifact_id=str(artifact_model.artifact_id),  # Casting to string
                file_name=artifact_model.file_name,
                agent_created=artifact_model.agent_created,
                uri=artifact_model.uri,
            )
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

    async def list_artifacts(self, task_id: str) -> List[Artifact]:
        session = self.Session()
        artifacts = session.query(ArtifactModel).filter_by(task_id=task_id).all()
        return [
            Artifact(
                artifact_id=str(artifact.artifact_id),
                file_name=artifact.file_name,
                agent_created=artifact.agent_created,
                uri=artifact.uri,
            )
            for artifact in artifacts
        ]
