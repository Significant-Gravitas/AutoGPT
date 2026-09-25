import asyncio
import json
import logging
from urllib.parse import urlsplit

from websockets.asyncio.client import ClientConnection, connect

from backend.util.link_checkout.cdp_models import (
    Control,
    Frame,
    FrameContext,
    FrameTree,
    Response,
    Result,
)
from backend.util.link_checkout.models import BoundField, BrowserBinding, CheckoutPlan
from backend.util.link_checkout.network import NetworkDrain
from backend.util.link_checkout.refusals import (
    FIELDS_NOT_READY,
    NOT_CARD_FIELDS,
    CheckoutRefused,
)

_logger = logging.getLogger("link_checkout.cdp")
_logger.addHandler(logging.NullHandler())
_logger.propagate = False
_logger.disabled = True


# The autofill field names (the last token of ``autocomplete``) each card role
# must declare. Payment processors' card frames and merchants' own card forms
# set them so browsers can autofill; nothing else on a page uses them.
CARD_AUTOCOMPLETE = {
    "number": ["cc-number"],
    "cvc": ["cc-csc"],
    "expiry": ["cc-exp"],
    "exp_month": ["cc-exp-month"],
    "exp_year": ["cc-exp-year"],
}

_CHECK_CONTROL = r"""function(names) {
    const box = this.getBoundingClientRect();
    const style = getComputedStyle(this);
    if (!this.isConnected || box.width <= 0 || box.height <= 0 ||
        style.visibility !== 'visible' || style.display === 'none' ||
        this.disabled) return 'not_ready';
    if (names === null) {
        const button = this instanceof HTMLButtonElement ||
            (this instanceof HTMLInputElement &&
             ['submit', 'button', 'image'].includes(this.type));
        return button ? 'ok' : 'not_card_field';
    }
    if (!(this instanceof HTMLInputElement) || this.type === 'hidden' ||
        this.readOnly) return 'not_card_field';
    const tokens = (this.getAttribute('autocomplete') || '').trim().toLowerCase()
        .split(/\s+/);
    if (!names.includes(tokens[tokens.length - 1])) return 'not_card_field';
    return this.value === '' ? 'ok' : 'not_ready';
}"""


class CDP:
    def __init__(self, socket: ClientConnection):
        self.socket = socket
        self.sequence = 0
        self.network = NetworkDrain()

    async def call(self, method: str, params: dict, session: str = "") -> Result:
        self.sequence += 1
        message = {"id": self.sequence, "method": method, "params": params}
        if session:
            message["sessionId"] = session
        await self.socket.send(json.dumps(message))
        while True:
            response = Response.model_validate_json(await self.socket.recv())
            self.network.observe(response.method, response.sessionId, response.params)
            if response.id != self.sequence:
                continue
            if response.error or response.result.exceptionDetails:
                raise RuntimeError("Private browser operation failed")
            return response.result

    async def monitor_network(self, controls: dict[str, Control]) -> None:
        self.network = NetworkDrain()
        for session in {control.session for control in controls.values()}:
            await self.call("Network.enable", {}, session)

    async def drain_network(self) -> bool:
        self.network.start_wait()
        try:
            async with asyncio.timeout(25):
                while not self.network.settled():
                    try:
                        raw = await asyncio.wait_for(self.socket.recv(), 0.25)
                    except TimeoutError:
                        continue
                    event = Response.model_validate_json(raw)
                    self.network.observe(event.method, event.sessionId, event.params)
            return True
        except TimeoutError:
            return False

    async def bind(self, expected_url: str) -> BrowserBinding:
        targets = (await self.call("Target.getTargets", {})).targetInfos
        matches = [t for t in targets if t.type == "page" and t.url == expected_url]
        if len(matches) != 1:
            raise RuntimeError("Checkout must match exactly one open tab")
        return BrowserBinding(
            endpoint="", target_id=matches[0].targetId, url=expected_url
        )

    async def attach(self, binding: BrowserBinding) -> str:
        current = await self.bind(binding.url)
        if current.target_id != binding.target_id:
            raise RuntimeError("The approved checkout tab changed")
        return await self.attach_target(binding.target_id)

    async def attach_target(self, target_id: str) -> str:
        result = await self.call(
            "Target.attachToTarget", {"targetId": target_id, "flatten": True}
        )
        return result.sessionId

    async def frames(self, binding: BrowserBinding) -> list[FrameContext]:
        session = await self.attach(binding)
        tree = (await self.call("Page.getFrameTree", {}, session)).frameTree
        if tree is None:
            raise RuntimeError("Checkout frame unavailable")
        frames = {
            f.id: FrameContext(frame=f, target_id=binding.target_id, session=session)
            for f in flatten_frames(tree)
        }
        targets = (await self.call("Target.getTargets", {})).targetInfos
        pending: list[FrameContext] = []
        for target in targets:
            if target.type != "iframe":
                continue
            child_session = await self.attach_target(target.targetId)
            child_tree = (
                await self.call("Page.getFrameTree", {}, child_session)
            ).frameTree
            if child_tree:
                pending.extend(
                    FrameContext(
                        frame=f, target_id=target.targetId, session=child_session
                    )
                    for f in flatten_frames(child_tree)
                )
        for _ in range(len(pending)):
            for context in pending:
                if context.frame.id in frames or context.frame.parentId in frames:
                    frames[context.frame.id] = context
        return list(frames.values())

    async def controls(
        self, binding: BrowserBinding, plan: CheckoutPlan, *, capture: bool = False
    ) -> dict[str, Control]:
        frames = await self.frames(binding)
        controls: dict[str, Control] = {}
        for role, target in plan.payment_fields().items():
            if target is None:
                continue
            matches = [
                f
                for f in frames
                if f.frame.url == (target.frame_url or plan.checkout_url)
            ]
            if len(matches) != 1 or not matches[0].frame.loaderId:
                raise CheckoutRefused(FIELDS_NOT_READY)
            context = matches[0]
            control = await self.resolve_control(context, role, target.selector)
            pinned = control.binding
            if not capture and pinned not in binding.fields:
                raise RuntimeError(
                    "Payment document or field changed after preparation"
                )
            await self.check_control(control, role)
            controls[role] = control
        if len(
            {
                (c.binding.target_id, c.binding.backend_node_id)
                for c in controls.values()
            }
        ) != len(controls):
            raise RuntimeError("Payment roles must refer to distinct elements")
        return controls

    async def resolve_control(
        self, context: FrameContext, role: str, selector: str
    ) -> Control:
        world = await self.call(
            "Page.createIsolatedWorld",
            {"frameId": context.frame.id, "worldName": "autogpt-private-checkout"},
            context.session,
        )
        element = await self.call(
            "Runtime.callFunctionOn",
            {
                "functionDeclaration": "function(selector) { const nodes = document.querySelectorAll(selector); return nodes.length === 1 ? nodes[0] : null; }",
                "arguments": [{"value": selector}],
                "executionContextId": world.executionContextId,
            },
            context.session,
        )
        if element.result is None or not element.result.objectId:
            raise CheckoutRefused(FIELDS_NOT_READY)
        node = (
            await self.call(
                "DOM.describeNode",
                {"objectId": element.result.objectId},
                context.session,
            )
        ).node
        if node is None:
            raise RuntimeError("Payment field unavailable")
        pinned = BoundField(
            role=role,
            target_id=context.target_id,
            frame_id=context.frame.id,
            loader_id=context.frame.loaderId,
            backend_node_id=node.backendNodeId,
        )
        return Control(
            binding=pinned,
            session=context.session,
            object_id=element.result.objectId,
        )

    async def check_control(self, control: Control, role: str) -> None:
        """A card role must be an empty input that declares itself that card
        field, so no selector can aim the card at an address, note or search
        box; the pay role must be a button."""
        result = await self.call(
            "Runtime.callFunctionOn",
            {
                "functionDeclaration": _CHECK_CONTROL,
                "objectId": control.object_id,
                "arguments": [{"value": CARD_AUTOCOMPLETE.get(role)}],
                "returnByValue": True,
            },
            control.session,
        )
        verdict = result.result.value if result.result else None
        if verdict == "not_card_field":
            raise CheckoutRefused(NOT_CARD_FIELDS)
        if verdict != "ok":
            raise CheckoutRefused(FIELDS_NOT_READY)


def flatten_frames(tree: FrameTree) -> list[Frame]:
    return [
        tree.frame,
        *[frame for child in tree.childFrames for frame in flatten_frames(child)],
    ]


def connection(endpoint: str):
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "ws"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username
        or parsed.password
    ):
        raise RuntimeError("Private browser requires a local CDP endpoint")
    return connect(endpoint, logger=_logger, max_size=1_000_000, open_timeout=5)


async def prepare_browser(endpoint: str, plan: CheckoutPlan) -> BrowserBinding:
    async with connection(endpoint) as socket:
        cdp = CDP(socket)
        binding = await cdp.bind(plan.checkout_url)
        controls = await cdp.controls(binding, plan, capture=True)
        binding.fields = [control.binding for control in controls.values()]
        binding.endpoint = endpoint
        return binding
