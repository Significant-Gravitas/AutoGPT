from __future__ import annotations

import json
import uuid
from pathlib import Path

from AFAAS.interfaces.db.db import AbstractMemory


class SQLLikeJSONFileMemory(AbstractMemory):
    def __init__(self, config: dict):
        raise NotImplementedError("SQLLikeJSONFileMemory")
        self._json_file_path = config.json_file_path

    async def connect(self, *kwargs):
        pass

    async def _load_file(self, table_name: str):
        file = Path(self._json_file_path / f"{table_name}.json")
        if file.is_file():
            with file.open() as f:
                data = json.load(f)
        else:
            data = {}
        return data

    async def _save_file(
        self,
        table_name: str,
        data: dict,
    ):
        file = Path(self._json_file_path / f"{table_name}.json")
        with file.open("w") as f:
            json.dump(data, f)

    async def get(self, key: uuid.UUID, table_name: str):
        data = await self._load_file(table_name=table_name)
        return data.get(key)

    async def add(self, key: uuid.UUID, value: dict, table_name: str):
        data = await self._load_file(table_name=table_name)
        data[key] = value
        await self._save_file(table_name=table_name, data=value)

    async def update(self, key: uuid.UUID, value: dict, table_name: str):
        file = Path(self._json_file_path / f"{table_name}.json")
        data = await self._load_file(table_name=table_name)
        if key in data:
            data[key] = value
            await self._save_file(table_name=table_name, data=data)
        else:
            raise KeyError(f"No such key '{key}' in file {file}")

    async def delete(self, key: uuid.UUID, table_name: str):
        file = Path(self._json_file_path / f"{table_name}.json")
        data = await self._load_file(table_name=table_name)
        if key in data:
            del data[key]
            await self._save_file(table_name=table_name, data=data)
        else:
            raise KeyError(f"No such key '{key}' in file {file}")

    async def list(self, table_name: str) -> dict:
        data = await self._load_file(table_name=table_name) > 0
        if len(data):
            return
        else:
            data = {}
