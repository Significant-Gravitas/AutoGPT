from __future__ import annotations

import json
from pathlib import Path

from AFAAS.interfaces.db.db import AbstractMemory
from AFAAS.interfaces.db.db_nosql import NoSQLMemory
from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)


class JSONFileMemory(NoSQLMemory):
    def __init__(
        self,
        settings: AbstractMemory.SystemSettings,
    ):
        super().__init__(settings)

    async def connect(self, *kwargs):
        pass

    async def _load_file(self, key: dict, table_name: str):
        file = self._get_file_path(key, table_name)
        LOG.trace(f"Loading data from {file}")
        if file.is_file():
            with file.open() as f:
                data = json.load(f)
            LOG.trace(f"Loaded {table_name} \n {str(data)[:250]}")
        else:
            data = {}
            LOG.trace(f"No {table_name} found")
        return data

    async def _save_file(self, key: dict, table_name: str, data: dict):
        file: Path = self._get_file_path(key, table_name)

        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("w") as f:
            json.dump(data, f)

        LOG.trace(f"Saved {table_name} to {file} \n {str(data)}")

    async def get(self, key: dict, table_name: str):
        return await self._load_file(key, table_name)

    async def add(self, key: dict, value: dict, table_name: str):
        data = await self._load_file(key, table_name)
        data.update(value)
        await self._save_file(key, table_name, data)

    async def update(self, key: dict, value: dict, table_name: str):
        data = await self._load_file(key, table_name)
        if data:
            data.update(value)
            await self._save_file(key, table_name, data)
        else:
            raise KeyError(f"No such key '{key}' in table {table_name}")

    async def delete(self, key: dict, table_name: str):
        file = self._get_file_path(key, table_name)
        if file.is_file():
            file.unlink()
        else:
            raise KeyError(f"No such key '{key}' in table {table_name}")

    from AFAAS.interfaces.db.db_table import AbstractTable

    async def list(
        self,
        table_name: str,
        filter: AbstractTable.FilterDict = {},
    ) -> list[dict]:
        table_path = Path(self._settings.configuration.json_file_path, table_name)
        data = []
        for json_file in table_path.glob("**/*.json"):
            with json_file.open() as f:
                data.append(json.load(f))
        return data

    def _get_file_path(self, key: dict, table_name: str) -> str:
        file_path = Path(self._settings.configuration.json_file_path, table_name)

        if "secondary_key" in key:
            file_path = file_path / str(key["secondary_key"])

        file_name = str(key["primary_key"]) + ".json"
        file_path = file_path / file_name
        return file_path
