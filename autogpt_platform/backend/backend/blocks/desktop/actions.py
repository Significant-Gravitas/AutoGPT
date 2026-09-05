import asyncio
import time
from enum import Enum
from typing import Literal, Optional

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.desktop._api import DesktopSession
from backend.blocks.desktop._common import (
    CREDENTIALS_FIELD_DESCRIPTION,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
)
from backend.blocks.desktop._cost import CostMeter, build_cost_meter
from backend.data.execution import ExecutionContext
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName
from backend.util.file import store_media_file
from backend.util.type import MediaFileType


class DesktopAction(str, Enum):
    SCREENSHOT = "screenshot"
    LEFT_CLICK = "left_click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    MIDDLE_CLICK = "middle_click"
    MOVE = "move"
    DRAG = "drag"
    SCROLL = "scroll"
    TYPE = "type"
    PRESS = "press"
    WAIT = "wait"


_CLICK_BUTTONS = {
    DesktopAction.LEFT_CLICK: 1,
    DesktopAction.MIDDLE_CLICK: 2,
    DesktopAction.RIGHT_CLICK: 3,
    DesktopAction.DOUBLE_CLICK: 1,
}


class DesktopActionBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.E2B], Literal["api_key"]
        ] = CredentialsField(description=CREDENTIALS_FIELD_DESCRIPTION)
        sandbox_id: str = SchemaField(description="ID of the desktop sandbox to act on")
        action: DesktopAction = SchemaField(
            description="The computer-use action to perform",
            default=DesktopAction.SCREENSHOT,
            advanced=False,
        )
        x: Optional[int] = SchemaField(
            description="X coordinate for click/move/drag-start actions",
            default=None,
            advanced=False,
        )
        y: Optional[int] = SchemaField(
            description="Y coordinate for click/move/drag-start actions",
            default=None,
            advanced=False,
        )
        to_x: Optional[int] = SchemaField(
            description="Destination X coordinate for drag", default=None
        )
        to_y: Optional[int] = SchemaField(
            description="Destination Y coordinate for drag", default=None
        )
        text: str = SchemaField(
            description="Text to type (for the 'type' action)", default=""
        )
        keys: list[str] = SchemaField(
            description=(
                "Keys to press together for the 'press' action, "
                "e.g. ['enter'] or ['ctrl', 'c']"
            ),
            default_factory=list,
        )
        scroll_direction: Literal["up", "down"] = SchemaField(
            description="Scroll direction (for the 'scroll' action)", default="down"
        )
        scroll_amount: int = SchemaField(
            description="Scroll clicks (for the 'scroll' action)", default=3
        )
        seconds: float = SchemaField(
            description="Seconds to wait (for the 'wait' action)", default=1.0
        )
        screenshot_after: bool = SchemaField(
            description="Capture a screenshot of the desktop after the action",
            default=True,
        )

    class Output(BlockSchemaOutput):
        result: str = SchemaField(description="Description of the performed action")
        screenshot: MediaFileType = SchemaField(
            description="Screenshot of the desktop after the action"
        )
        cost_meter: CostMeter = SchemaField(
            description="Estimated provider cost telemetry for this block run"
        )

    def __init__(self):
        super().__init__(
            id="a1e2b001-0002-4000-8000-de5c704b0002",
            description=(
                "Performs a computer-use action (mouse, keyboard, scroll, "
                "screenshot) on an interactive desktop sandbox."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=DesktopActionBlock.Input,
            output_schema=DesktopActionBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "sandbox_id": "test-sandbox-id",
                "action": DesktopAction.MOVE.value,
                "x": 10,
                "y": 20,
                "screenshot_after": False,
            },
            test_output=[
                ("result", "Performed move"),
                ("cost_meter", lambda v: v["provider"] == "e2b"),
            ],
            test_mock={
                "perform_action": lambda *args, **kwargs: ("Performed move", None)
            },
        )

    async def perform_action(
        self, api_key: str, input_data: Input
    ) -> tuple[str, Optional[str]]:
        session = await DesktopSession.connect(input_data.sandbox_id, api_key)
        await self._dispatch(session, input_data)
        screenshot = None
        if input_data.screenshot_after or input_data.action == DesktopAction.SCREENSHOT:
            screenshot = await session.screenshot_base64()
        return f"Performed {input_data.action.value}", screenshot

    async def _dispatch(self, session: DesktopSession, input_data: Input) -> None:
        action = input_data.action
        if action == DesktopAction.SCREENSHOT:
            return
        if action in _CLICK_BUTTONS:
            await session.click(
                button=_CLICK_BUTTONS[action],
                x=input_data.x,
                y=input_data.y,
                double=action == DesktopAction.DOUBLE_CLICK,
            )
        elif action == DesktopAction.MOVE:
            x, y = _required_coords(input_data.x, input_data.y, "move")
            await session.move_mouse(x, y)
        elif action == DesktopAction.DRAG:
            from_x, from_y = _required_coords(input_data.x, input_data.y, "drag")
            to_x, to_y = _required_coords(
                input_data.to_x, input_data.to_y, "drag destination"
            )
            await session.drag(from_x, from_y, to_x, to_y)
        elif action == DesktopAction.SCROLL:
            await session.scroll(input_data.scroll_direction, input_data.scroll_amount)
        elif action == DesktopAction.TYPE:
            await session.type_text(input_data.text)
        elif action == DesktopAction.PRESS:
            await session.press(input_data.keys)
        elif action == DesktopAction.WAIT:
            await asyncio.sleep(min(input_data.seconds, 60))

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        start = time.monotonic()
        try:
            result, screenshot_b64 = await self.perform_action(
                credentials.api_key.get_secret_value(), input_data
            )
            yield "result", result
            if screenshot_b64:
                stored = await store_media_file(
                    file=MediaFileType(f"data:image/png;base64,{screenshot_b64}"),
                    execution_context=execution_context,
                    return_format="for_block_output",
                )
                yield "screenshot", stored
            yield "cost_meter", build_cost_meter(
                input_data.sandbox_id, time.monotonic() - start
            ).model_dump()
        except Exception as e:
            yield "error", str(e)


def _required_coords(
    x: Optional[int], y: Optional[int], action: str
) -> tuple[int, int]:
    if x is None or y is None:
        raise ValueError(f"x and y coordinates are required for {action}")
    return x, y
