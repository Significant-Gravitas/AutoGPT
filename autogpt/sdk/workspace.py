import abc
import os
import typing
from pathlib import Path

import aiohttp
from fastapi import Response


class Workspace(abc.ABC):
    @abc.abstractclassmethod
    def __init__(self, base_path: str) -> None:
        self.base_path = base_path

    @abc.abstractclassmethod
    def read(self, task_id: str, path: str) -> bytes:
        pass

    @abc.abstractclassmethod
    def write(self, task_id: str, path: str, data: bytes) -> None:
        pass

    @abc.abstractclassmethod
    def delete(
        self, task_id: str, path: str, directory: bool = False, recursive: bool = False
    ) -> None:
        pass

    @abc.abstractclassmethod
    def exists(self, task_id: str, path: str) -> bool:
        pass

    @abc.abstractclassmethod
    def list(self, task_id: str, path: str) -> typing.List[str]:
        pass


class LocalWorkspace(Workspace):
    def __init__(self, base_path: str):
        self.base_path = Path(base_path).resolve()

    def _resolve_path(self, task_id: str, path: str) -> Path:
        abs_path = (self.base_path / task_id / path).resolve()
        if not str(abs_path).startswith(str(self.base_path)):
            raise ValueError("Directory traversal is not allowed!")
        (self.base_path / task_id).mkdir(parents=True, exist_ok=True)
        return abs_path

    def read(self, task_id: str, path: str) -> bytes:
        path = self.base_path / task_id / path
        with open(self._resolve_path(task_id, path), "rb") as f:
            return f.read()

    def write(self, task_id: str, path: str, data: bytes) -> None:
        path = self.base_path / task_id / path
        with open(self._resolve_path(task_id, path), "wb") as f:
            f.write(data)

    def delete(
        self, task_id: str, path: str, directory: bool = False, recursive: bool = False
    ) -> None:
        path = self.base_path / task_id / path
        resolved_path = self._resolve_path(task_id, path)
        if directory:
            if recursive:
                os.rmdir(resolved_path)
            else:
                os.removedirs(resolved_path)
        else:
            os.remove(resolved_path)

    def exists(self, task_id: str, path: str) -> bool:
        path = self.base_path / task_id / path
        return self._resolve_path(task_id, path).exists()

    def list(self, task_id: str, path: str) -> typing.List[str]:
        path = self.base_path / task_id / path
        base = self._resolve_path(task_id, path)
        return [str(p.relative_to(self.base_path / task_id)) for p in base.iterdir()]


async def load_from_uri(self, uri: str, task_id: str, workspace: Workspace) -> bytes:
    """
    Load file from given URI and return its bytes.
    """
    file_path = None
    try:
        if uri.startswith("file://"):
            file_path = uri.split("file://")[1]
            if not workspace.exists(task_id, file_path):
                return Response(status_code=500, content="File not found")
            return workspace.read(task_id, file_path)
        elif uri.startswith("http://") or uri.startswith("https://"):
            async with aiohttp.ClientSession() as session:
                async with session.get(uri) as resp:
                    if resp.status != 200:
                        return Response(
                            status_code=500, content="Unable to load from URL"
                        )
                    return await resp.read()
        else:
            return Response(status_code=500, content="Loading from unsupported uri")
    except Exception as e:
        return Response(status_code=500, content=str(e))
