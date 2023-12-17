from __future__ import annotations

import abc
import enum
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

from typing_extensions import TypedDict

from AFAAS.interfaces.prompts.strategy import AbstractPromptStrategy

from .features.agentmixin import AgentMixin

if TYPE_CHECKING:
    from AFAAS.interfaces.adapters.chat_model import \
        ChatModelResponse

    from . import BaseAgent
from AFAAS.lib.sdk.logger import AFAASLogger
LOG = AFAASLogger(name = __name__)

class BaseLoopMeta(abc.ABCMeta):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        instance.__init__(*args, **kwargs)
        return instance


class BaseLoopHookKwargs(TypedDict):
    agent: BaseAgent
    loop: BaseLoop
    user_input_handler: Callable[[str], Awaitable[str]]
    user_message_handler: Callable[[str], Awaitable[str]]
    kwargs: dict[str, Any]


class BaseLoopHook(TypedDict):
    name: str
    function: Callable
    kwargs: BaseLoopHookKwargs
    expected_return: Any
    callback_function: Optional[Callable[..., BaseLoopHookKwargs]]


class UserFeedback(str, enum.Enum):
    """Enum for user feedback."""

    AUTHORIZE = "GENERATE NEXT COMMAND JSON"
    EXIT = "EXIT"
    TEXT = "TEXT"


class BaseLoop(AgentMixin, abc.ABC, metaclass=BaseLoopMeta):
    class LoophooksDict(TypedDict):
        begin_run: dict[BaseLoopHook]
        end_run: dict[BaseLoopHook]

    _loophooks: LoophooksDict

    @abc.abstractmethod
    def __init__(self):
        # Step 1 : Setting loop variables
        self._is_running: bool = False
        self._loophooks = self.LoophooksDict()
        for key in self.LoophooksDict.__annotations__.keys():
            self._loophooks[key] = {}

        # Step 2 : Setting task variables
        self._task_queue = []
        self._completed_tasks = []
        self._current_task = None
        self._next_tool = None

        # Setting default handlers
        self._user_input_handler: Optional[Callable[[str], Awaitable[str]]] = None
        self._user_message_handler: Optional[Callable[[str], Awaitable[str]]] = None

    @abc.abstractmethod
    async def run(
        self,
        agent: BaseAgent,
        hooks: LoophooksDict,
        user_input_handler: Optional[Callable[[str], Awaitable[str]]] = None,
        user_message_handler: Optional[Callable[[str], Awaitable[str]]] = None,
    ) -> None | dict:
        ...

    async def handle_hooks(
        self,
        hook_key: str,
        hooks: LoophooksDict,
        agent: BaseAgent,
        user_input_handler: Optional[Callable[[str], Awaitable[str]]] = None,
        user_message_handler: Optional[Callable[[str], Awaitable[str]]] = None,
    ):
        if isinstance(user_input_handler, Callable) and user_input_handler is not None:
            self._user_input_handler = user_input_handler
        elif self._user_input_handler is None:
            raise TypeError(
                "`user_input_handler` must be a callable or set previously."
            )

        if (
            isinstance(user_message_handler, Callable)
            and user_message_handler is not None
        ):
            self._user_message_handler = user_message_handler
        elif self._user_message_handler is None:
            raise TypeError(
                "`user_message_handler` must be a callable or set previously."
            )

        if self._loophooks.get(hook_key):
            for key, hook in self._loophooks[hook_key].items():
                # if isinstance(hook, BaseLoopHook):
                LOG.trace(f"Executing hook {key}")
                LOG.debug(f"Hook class is {hook.__class__}'")
                await self.execute_hook(hook=hook, agent=agent)
                # else :
                #     raise TypeError(f"Hook {key} is not a BaseLoopHook but is a {hook.__class__}")

    async def execute_hook(self, hook: BaseLoopHook, agent: BaseAgent):
        user_input_handler = self._user_input_handler
        user_message_handler = self._user_message_handler
        kwargs: BaseLoopHookKwargs = {
            "agent": agent,
            "loop": self,
            "user_input_handler": user_input_handler,
            "user_message_handler": user_message_handler,
            "kwargs": hook["kwargs"],
        }

        result = hook["function"](**kwargs)

        if result != hook["expected_return"]:
            if hook["callback_function"] is not None:
                kwargs["result"] = result
                hook["callback_function"](**kwargs)

    async def start(
        self,
        agent: BaseAgent = None,
        user_input_handler: Optional[Callable[[str], Awaitable[str]]] = None,
        user_message_handler: Optional[Callable[[str], Awaitable[str]]] = None,
    ) -> None:
        if isinstance(user_input_handler, Callable) and user_input_handler is not None:
            self._user_input_handler = user_input_handler
        elif self._user_input_handler is None:
            raise TypeError(
                "`user_input_handler` must be a callable or set previously."
            )

        if (
            isinstance(user_message_handler, Callable)
            and user_message_handler is not None
        ):
            self._user_message_handler = user_message_handler
        elif self._user_message_handler is None:
            raise TypeError(
                "`user_message_handler` must be a callable or set previously."
            )

        LOG.trace("Starting loop")
        self._active = True

    async def stop(
        self,
        agent: BaseAgent = None,
        user_input_handler: Optional[Callable[[str], Awaitable[str]]] = None,
        user_message_handler: Optional[Callable[[str], Awaitable[str]]] = None,
    ) -> None:
        if isinstance(user_input_handler, Callable) and user_input_handler is not None:
            self._user_input_handler = user_input_handler
        elif self._user_input_handler is None:
            raise TypeError(
                "`user_input_handler` must be a callable or set previously."
            )

        if (
            isinstance(user_message_handler, Callable)
            and user_message_handler is not None
        ):
            self._user_message_handler = user_message_handler
        elif self._user_message_handler is None:
            raise TypeError(
                "`user_message_handler` must be a callable or set previously."
            )

        LOG.trace("Stoping loop")
        self._active = False

    def __repr__(self):
        return "BaseLoop()"

    #
    # SHORTCUTS !
    #


