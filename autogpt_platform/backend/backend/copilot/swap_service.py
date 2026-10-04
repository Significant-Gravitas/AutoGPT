"""The only backend service the credential swap proxy calls: the two swap
methods and nothing else.  Hosted beside ``DatabaseManager`` (``backend.db``);
see ``swap_proxy/README.md`` for why."""

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Optional

from backend.copilot.swap_credentials import (
    SwapCredential,
    get_swap_bindings,
    resolve_swap_credential,
)
from backend.data import db
from backend.util.service import AppService, UnhealthyServiceError, expose
from backend.util.settings import Config

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)


class SwapCredentialService(AppService):
    @classmethod
    def get_port(cls) -> int:
        return Config().swap_credential_service_port

    @asynccontextmanager
    async def lifespan(self, app: "FastAPI"):
        async with super().lifespan(app):
            await db.connect()
            logger.info(f"[{self.service_name}] ✅ Ready")
            yield
            await db.disconnect()

    async def health_check(self) -> str:
        if not db.is_connected():
            raise UnhealthyServiceError("Database is not connected")
        await db.query_raw_with_schema("SELECT 1 as health_check")
        return await super().health_check()

    @expose
    async def get_swap_bindings(self) -> dict[str, list[str]]:
        return await get_swap_bindings()

    @expose
    async def resolve_swap_credential(
        self, user_id: str, name: str, host: str
    ) -> Optional[SwapCredential]:
        return await resolve_swap_credential(user_id, name, host)
