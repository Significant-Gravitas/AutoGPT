import abc
import os
import typing
from pathlib import Path


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
        path = path if not path.startswith("/") else path[1:]
        abs_path = (self.base_path / task_id / path).resolve()
        abs_path_str = abs_path.as_posix()
        base_path_str = self.base_path.as_posix()
        if not abs_path_str.startswith(base_path_str):
            print("Error")
            raise ValueError(f"Directory traversal is not allowed! - {abs_path_str}")
        try:
            abs_path.parent.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            pass
        return abs_path

    def read(self, task_id: str, path: str) -> bytes:
        with open(self._resolve_path(task_id, path), "rb") as f:
            return f.read()

    def write(self, task_id: str, path: str, data: bytes) -> None:
        file_path = self._resolve_path(task_id, path)
        with open(file_path, "wb") as f:
            f.write(data)

    def write_str(self, task_id: str, path: str, data: str) -> None:
        file_path = self._resolve_path(task_id, path)
        with open(file_path, 'w') as f:
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

    def get_cwd_path(self, task_id: str) -> str:
        path_obj = (self.base_path / task_id).resolve()
        return path_obj.as_posix()
    
    def get_temp_path(self, task_id: str) -> str:
        path_obj = (self.base_path.parent / "temp_folder")
        return path_obj.as_posix()
