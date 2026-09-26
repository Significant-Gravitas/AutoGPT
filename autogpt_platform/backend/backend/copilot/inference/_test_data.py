"""What the ``structured_complete`` tests share: a response model, a sync
route on any provider, the platform key a transport would give, and the
provider responses and errors the Anthropic path sees."""

import anthropic
import httpx
from pydantic import BaseModel

from backend.copilot.dream.structured_output import output_tool_name
from backend.copilot.transport_routing import ProviderRoutingKwargs
from backend.util.llm.conversions import ToolCall, ToolContentBlock
from backend.util.llm.providers import ProviderResponse

from .context import InferenceContext, InferenceJob, InferenceScope, RouteDecision

# Where the tests patch the transport's key and the provider call.
ROUTING = "backend.copilot.inference.routing.routing_kwargs_for_chat_transport"
CALL_PROVIDER = "backend.copilot.inference.complete.call_provider"


class SampleFact(BaseModel):
    content: str
    confidence: float


class SampleOutput(BaseModel):
    facts: list[SampleFact]


MESSAGES = [{"role": "user", "content": "give me a fact"}]
ONE_FACT = '{"facts": [{"content": "x", "confidence": 0.9}]}'

# Anthropic's 400 for a forced ``tool_choice`` on a model that takes none.
FORCED_TOOL_REJECTION = (
    'tool_choice: type "tool" and "any" are not supported for this model.'
)


def platform_routing(
    provider="open_router", api_key="sk-or-test", base_url=None
) -> ProviderRoutingKwargs:
    return ProviderRoutingKwargs(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        supports_flex=provider == "open_router",
        cost_log_provider={"open_router": "open_router", "ollama": "ollama"}.get(
            provider, "anthropic"
        ),
    )


ANTHROPIC = platform_routing("anthropic", "sk-ant-test")


def sync_ctx(
    provider="open_router",
    model="anthropic/claude-sonnet-4-6",
    *,
    timeout_seconds: float | None = None,
    payer="platform_allowance",
) -> InferenceContext:
    return InferenceContext(
        scope=InferenceScope(user_id="u1"),
        job=InferenceJob(
            kind="dream",
            phase="consolidate",
            correlation_id="pass-1",
            latency_class="deferred",
            tier="standard",
            timeout_seconds=timeout_seconds,
        ),
        route=RouteDecision(
            engine="provider_sync",
            auth_provider="platform",
            provider=provider,
            model=model,
            payer=payer,
            execution_path="sync_baseline",
            cost_log_provider="test",
            reason="test",
        ),
    )


def tool_response(arguments: str, **usage: int) -> ProviderResponse:
    """What ``call_provider`` returns for a forced tool call: ``content``
    is the tool NAME, the JSON is in the tool call's arguments."""
    name = output_tool_name(SampleOutput)
    return ProviderResponse(
        content=name,
        prompt_tokens=usage.get("prompt_tokens", 1),
        completion_tokens=usage.get("completion_tokens", 1),
        cache_read_tokens=usage.get("cache_read_tokens", 0),
        tool_calls=[
            ToolContentBlock(
                id="toolu_1",
                type="tool_use",
                function=ToolCall(name=name, arguments=arguments),
            )
        ],
    )


def bad_request(message: str) -> anthropic.BadRequestError:
    response = httpx.Response(
        400, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    return anthropic.BadRequestError(message, response=response, body=None)


def server_error(message: str) -> anthropic.InternalServerError:
    response = httpx.Response(
        500, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    return anthropic.InternalServerError(message, response=response, body=None)
