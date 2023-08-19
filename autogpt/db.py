"""
This is an example implementation of the Agent Protocol DB for development Purposes
It uses SQLite as the database and file store backend.
IT IS NOT ADVISED TO USE THIS IN PRODUCTION!
"""

from typing import Dict, List, Optional

from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, joinedload, relationship, sessionmaker

from .schema import Artifact, Status, Step, Task, TaskInput


class Base(DeclarativeBase):
    pass


class DataNotFoundError(Exception):
    pass


class TaskModel(Base):
    __tablename__ = "tasks"

    task_id = Column(Integer, primary_key=True, autoincrement=True)
    input = Column(String)
    additional_input = Column(String)

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


def convert_to_task(task_obj: TaskModel) -> Task:
    return Task(
        task_id=task_obj.task_id,
        input=task_obj.input,
        additional_input=task_obj.additional_input,
        artifacts=[],
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
    def __init__(self, database_string) -> None:
        super().__init__()
        self.engine = create_engine(database_string)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        print("Databases Created")

    async def create_task(self, input: Optional[str], additional_input: Optional[TaskInput] = None) -> Task:
        with self.Session() as session:
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
        try:
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
        except Exception as e:
            print(e)
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
        return Artifact(
            artifact_id=str(new_artifact.artifact_id),
            file_name=new_artifact.file_name,
            agent_created=new_artifact.agent_created,
            uri=new_artifact.uri,
        )

    async def get_task(self, task_id: int) -> Task:
        """Get a task by its id"""
        session = self.Session()
        if task_obj := (
            session.query(TaskModel)
            .options(joinedload(TaskModel.artifacts))
            .filter_by(task_id=task_id)
            .first()
        ):
            return convert_to_task(task_obj)
        else:
            raise DataNotFoundError("Task not found")

    async def get_step(self, task_id: int, step_id: int) -> Step:
        session = self.Session()
        if step := (
            session.query(StepModel)
            .options(joinedload(StepModel.artifacts))
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
                artifacts=[],
                steps=[],
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
        with self.Session() as session:
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
