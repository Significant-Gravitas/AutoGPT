from __future__ import annotations

import json
from logging import Logger
from pathlib import Path
from typing import List

from AFAAS.app.core.memory.base import AbstractMemory
from AFAAS.app.core.memory.nosqlmemory import NoSQLMemory


class JSONFileMemory(NoSQLMemory):
    def __init__(
        self,
        settings: AbstractMemory.SystemSettings,
        logger: Logger,
    ):
        super().__init__(settings, logger)

    def connect(self, *kwargs):
        pass

    def _load_file(self, key: dict, table_name: str):
        file = self._get_file_path(key, table_name)
        self._logger.debug(f"Loading data from {file}")
        if file.is_file():
            with file.open() as f:
                data = json.load(f)
            self._logger.debug(f"Loaded {table_name} \n {str(data)[:250]}")
        else:
            data = {}
            self._logger.debug(f"No {table_name} found")
        return data

    def _save_file(self, key: dict, table_name: str, data: dict):
        file: Path = self._get_file_path(key, table_name)

        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("w") as f:
            json.dump(data, f)

        self._logger.debug(f"Saved {table_name} to {file} \n {str(data)}")

    def get(self, key: dict, table_name: str):
        return self._load_file(key, table_name)

    def add(self, key: dict, value: dict, table_name: str):
        data = self._load_file(key, table_name)
        data.update(value)
        self._save_file(key, table_name, data)

    def update(self, key: dict, value: dict, table_name: str):
        data = self._load_file(key, table_name)
        if data:
            data.update(value)
            self._save_file(key, table_name, data)
        else:
            raise KeyError(f"No such key '{key}' in table {table_name}")

    def delete(self, key: dict, table_name: str):
        file = self._get_file_path(key, table_name)
        if file.is_file():
            file.unlink()
        else:
            raise KeyError(f"No such key '{key}' in table {table_name}")

    def list(self, table_name: str) -> list[dict]:
        table_path = Path(self._configuration.json_file_path, table_name)
        data = []
        for json_file in table_path.glob("**/*.json"):
            with json_file.open() as f:
                data.append(json.load(f))
        return data

    def _get_file_path(self, key: dict, table_name: str) -> str:
        file_path = Path(self._configuration.json_file_path, table_name)

        if "secondary_key" in key:
            file_path = file_path / str(key["secondary_key"])

        file_name = str(key["primary_key"]) + ".json"
        file_path = file_path / file_name
        return file_path
