"""
This is an example implementation of the Agent Protocol DB for development Purposes
It uses SQlite as the database and file store backend.
IT IS NOT ADVISED TO USE THIS IN PRODUCTION!
"""


import sqlite3
from typing import Dict, List, Optional

from agent_protocol import Artifact, Step, Task, TaskDB
from agent_protocol.models import Status, TaskInput


class DataNotFoundError(Exception):
    pass


class AgentDB(TaskDB):
    def __init__(self, database_name) -> None:
        super().__init__()
        self.conn = sqlite3.connect(database_name)
        cursor = self.conn.cursor()

        # Create tasks table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS tasks (
            task_id INTEGER PRIMARY KEY AUTOINCREMENT,
            input TEXT,
            additional_input TEXT
        )
        """
        )

        # Create steps table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS steps (
            step_id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER,
            name TEXT,
            status TEXT,
            is_last INTEGER DEFAULT 0,
            additional_properties TEXT,
            FOREIGN KEY (task_id) REFERENCES tasks(task_id)
        )
        """
        )

        # Create artifacts table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER,
            step_id INTEGER,
            file_name TEXT,
            relative_path TEXT,
            file_data BLOB,
            FOREIGN KEY (task_id) REFERENCES tasks(task_id)
        )
        """
        )

        # Commit the changes
        self.conn.commit()
        print("Databases Created")

    async def create_task(
        self,
        input: Optional[str],
        additional_input: Optional[TaskInput] = None,
        artifacts: List[Artifact] = None,
        steps: List[Step] = None,
    ) -> Task:
        """Create a task"""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO tasks (input, additional_input) VALUES (?, ?)",
            (input, additional_input.json() if additional_input else None),
        )
        task_id = cursor.lastrowid
        self.conn.commit()
        if task_id:
            return await self.get_task(task_id)
        else:
            raise DataNotFoundError("Task not found")

    async def create_step(
        self,
        task_id: str,
        name: Optional[str] = None,
        is_last: bool = False,
        additional_properties: Optional[Dict[str, str]] = None,
    ) -> Step:
        """Create a step for a given task"""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO steps (task_id, name, status, is_last, additional_properties) VALUES (?, ?, ?, ?, ?)",
            (task_id, name, "created", is_last, additional_properties),
        )
        step_id = cursor.lastrowid
        self.conn.commit()
        if step_id and task_id:
            return await self.get_step(task_id, step_id)
        else:
            raise DataNotFoundError("Step not found")

    async def create_artifact(
        self,
        task_id: str,
        file_name: str,
        relative_path: Optional[str] = None,
        step_id: Optional[str] = None,
        file_data: bytes | None = None,
    ) -> Artifact:
        """Create an artifact for a given task"""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO artifacts (task_id, step_id, file_name, relative_path, file_data) VALUES (?, ?, ?, ?, ?)",
            (task_id, step_id, file_name, relative_path, file_data),
        )
        artifact_id = cursor.lastrowid
        self.conn.commit()
        return await self.get_artifact(task_id, artifact_id)

    async def get_task(self, task_id: int) -> Task:
        """Get a task by its id"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,))
        if task := cursor.fetchone():
            task = Task(task_id=task[0], input=task[1], additional_input=task[2])
            cursor.execute("SELECT * FROM steps WHERE task_id=?", (task_id,))
            steps = cursor.fetchall()
            if steps:
                for step in steps:
                    status = (
                        Status.created if step[3] == "created" else Status.completed
                    )
                    task.steps.append(
                        Step(
                            task_id=step[1],
                            step_id=step[0],
                            name=step[2],
                            status=status,
                            is_last=True if step[4] == 1 else False,
                            additional_properties=step[5],
                        )
                    )
            # print(f"Getting task {task_id}.... Task details: {task}")
            return task
        else:
            raise DataNotFoundError("Task not found")

    async def get_step(self, task_id: int, step_id: int) -> Step:
        """Get a step by its id"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM steps WHERE task_id=? AND step_id=?", (task_id, step_id)
        )
        if step := cursor.fetchone():
            return Step(
                task_id=task_id,
                step_id=step_id,
                name=step[2],
                status=step[3],
                is_last=step[4] == 1,
                additional_properties=step[5],
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
        """Update a step by its id"""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE steps SET status=?, additional_properties=? WHERE task_id=? AND step_id=?",
            (status, additional_properties, task_id, step_id),
        )
        self.conn.commit()
        return await self.get_step(task_id, step_id)

    async def get_artifact(self, task_id: str, artifact_id: str) -> Artifact:
        """Get an artifact by its id"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT artifact_id, file_name, relative_path FROM artifacts WHERE task_id=? AND artifact_id=?",
            (task_id, artifact_id),
        )
        if artifact := cursor.fetchone():
            return Artifact(
                artifact_id=artifact[0],
                file_name=artifact[1],
                relative_path=artifact[2],
            )
        else:
            raise DataNotFoundError("Artifact not found")

    async def get_artifact_file(self, task_id: str, artifact_id: str) -> bytes:
        """Get an artifact file by its id"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT file_data, file_name FROM artifacts WHERE task_id=? AND artifact_id=?",
            (task_id, artifact_id),
        )
        if artifact := cursor.fetchone():
            return artifact[0]

    async def list_tasks(self) -> List[Task]:
        """List all tasks"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM tasks")
        tasks = cursor.fetchall()
        return [
            Task(task_id=task[0], input=task[1], additional_input=task[2])
            for task in tasks
        ]

    async def list_steps(self, task_id: str) -> List[Step]:
        """List all steps for a given task"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM steps WHERE task_id=?", (task_id,))
        steps = cursor.fetchall()
        return [
            Step(task_id=task_id, step_id=step[0], name=step[2], status=step[3])
            for step in steps
        ]
