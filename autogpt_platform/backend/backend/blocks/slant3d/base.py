import asyncio
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryFile
from urllib.parse import quote

from backend.blocks._base import Block
from backend.data.execution import ExecutionContext
from backend.util.file import get_exec_file_path, store_media_file
from backend.util.request import Requests
from backend.util.type import MediaFileType

from ._api import CustomerDetails, OrderItem, Profile


class Slant3DBlockBase(Block):
    BASE_URL = "https://slant3dapi.com/v2/api"

    def _get_headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def _make_request(
        self, method: str, endpoint: str, api_key: str, **kwargs
    ) -> dict:
        response = await Requests(
            raise_for_status=False, retry_max_attempts=3 if method == "GET" else 1
        ).request(
            method=method,
            url=f"{self.BASE_URL}/{endpoint}",
            headers=self._get_headers(api_key),
            **kwargs,
        )
        try:
            result = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Slant3D returned an invalid response (HTTP {response.status})"
            ) from exc
        if not response.ok or result.get("success") is False:
            error = result.get("error") or result.get("message") or "Unknown error"
            match error:
                case {"message": str(message)}:
                    error = message
            raise RuntimeError(f"Slant3D API request failed: {error}")
        return result

    async def _resolve_platform_id(self, platform_id: str, api_key: str) -> str:
        if platform_id:
            return platform_id
        response = await self._make_request("GET", "platforms", api_key)
        platforms = [p for p in response["data"] if p.get("enabled", True)]
        if len(platforms) != 1:
            raise ValueError(
                "Set platform_id to an enabled Slant3D platform ID. "
                "Automatic selection requires exactly one enabled platform."
            )
        return platforms[0]["id"]

    async def _resolve_filament_id(
        self, profile: Profile, color: str, api_key: str
    ) -> str:
        response = await self._make_request(
            "GET",
            "filaments",
            api_key,
            params={"profile": profile.value, "color": color},
        )
        matches = [
            filament
            for filament in response["data"]
            if filament["profile"].casefold() == profile.value.casefold()
            and filament["color"].casefold() == color.casefold()
            and filament.get("available") is not False
        ]
        if len(matches) != 1:
            raise ValueError(
                f"No unique available filament for {profile.value} {color}. "
                "Choose a filament_id from the Slant3D Filament block."
            )
        return matches[0]["publicId"]

    async def _upload_file(
        self,
        file_url: MediaFileType,
        platform_id: str,
        api_key: str,
        *,
        execution_context: ExecutionContext,
        staging_lock: asyncio.Lock | None = None,
    ) -> str:
        graph_exec_id = execution_context.graph_exec_id
        if not graph_exec_id:
            raise ValueError("execution_context.graph_exec_id is required")
        with TemporaryFile() as content:
            async with staging_lock or asyncio.Lock():
                local_path = await store_media_file(
                    file=file_url,
                    execution_context=execution_context,
                    return_format="for_local_processing",
                )
                path = Path(get_exec_file_path(graph_exec_id, local_path))
                with path.open("rb") as source:
                    await asyncio.to_thread(copyfileobj, source, content)
            content.seek(0)
            upload = await self._make_request(
                "POST",
                "files/direct-upload",
                api_key,
                json={"name": path.name, "platformId": platform_id},
            )
            data = upload["data"]
            await Requests(retry_max_attempts=1).put(
                data["presignedUrl"],
                data=content,
                headers={"Content-Type": "application/octet-stream"},
            )
            confirmed = await self._make_request(
                "POST",
                "files/confirm-upload",
                api_key,
                json={"filePlaceholder": data["filePlaceholder"]},
            )
        return confirmed["data"]["publicFileServiceId"]

    async def _format_order_data(
        self,
        customer: CustomerDetails,
        order_number: str,
        items: list[OrderItem],
        api_key: str,
        platform_id: str = "",
        *,
        execution_context: ExecutionContext,
    ) -> dict:
        platform_id = await self._resolve_platform_id(platform_id, api_key)
        return {
            "customer": {
                "platformId": platform_id,
                "details": {
                    "email": customer.email,
                    "address": {
                        "name": customer.name,
                        "line1": customer.address,
                        "line2": customer.address_line2,
                        "city": customer.city,
                        "state": customer.state,
                        "zip": customer.zip,
                        "country": customer.country_iso,
                    },
                },
            },
            "items": await self._format_order_items(
                items, platform_id, api_key, execution_context=execution_context
            ),
            "metadata": {"orderNumber": order_number},
        }

    async def _format_order_items(
        self,
        items: list[OrderItem],
        platform_id: str,
        api_key: str,
        *,
        execution_context: ExecutionContext,
    ) -> list[dict]:
        filaments: dict[tuple[Profile, str], asyncio.Task[str]] = {}
        staging_lock = asyncio.Lock()
        concurrency = asyncio.Semaphore(4)

        async def prepare(item: OrderItem) -> dict:
            async with concurrency:
                filament_id = item.filament_id
                if not filament_id:
                    key = (item.profile, item.color.casefold())
                    if key not in filaments:
                        filaments[key] = asyncio.create_task(
                            self._resolve_filament_id(item.profile, item.color, api_key)
                        )
                    filament_id = await filaments[key]
                return await self._format_order_item(
                    item.model_copy(update={"filament_id": filament_id}),
                    platform_id,
                    api_key,
                    execution_context=execution_context,
                    staging_lock=staging_lock,
                )

        tasks = [asyncio.create_task(prepare(item)) for item in items]
        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _format_order_item(
        self,
        item: OrderItem,
        platform_id: str,
        api_key: str,
        *,
        execution_context: ExecutionContext,
        staging_lock: asyncio.Lock | None = None,
    ) -> dict:
        filament_id = item.filament_id or await self._resolve_filament_id(
            item.profile, item.color, api_key
        )
        file_id = item.file_id or await self._upload_file(
            item.file_url,
            platform_id,
            api_key,
            execution_context=execution_context,
            staging_lock=staging_lock,
        )
        return {
            "type": "PRINT",
            "publicFileServiceId": file_id,
            "filamentId": filament_id,
            "quantity": item.quantity,
        }

    async def _process_order(self, order_id: str, api_key: str) -> dict:
        return await self._make_request(
            "POST", f"orders/{quote(order_id, safe='')}", api_key
        )
