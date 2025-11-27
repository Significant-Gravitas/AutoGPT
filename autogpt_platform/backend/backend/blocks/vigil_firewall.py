from typing import Any, Dict, List

import requests

from backend.data.block import Block, BlockOutput, BlockSchemaInput, BlockSchemaOutput
from backend.data.model import SchemaField


class AgentShieldFirewallBlock(Block):
    """
    Block: AgentShield / Vigil Firewall + Analytics

    Routes LLM calls through the AgentShield gateway so that:
    - per-agent / per-tenant policies are enforced
    - PII / unsafe prompts can be blocked
    - every request/response is logged for analytics (Vigil)
    """

    class Input(BlockSchemaInput):
        gateway_url: str = SchemaField(
            description=(
                "Full URL for the AgentShield/Vigil gateway "
                "`/v1/chat/completions` endpoint."
            ),
            placeholder="https://gateway.example.com/v1/chat/completions",
        )
        api_key: str = SchemaField(
            description="AgentShield/Vigil API key for this agent.",
            secret=True,
        )
        tenant_id: str = SchemaField(
            description="Tenant / organization identifier (e.g. `demo-enterprise`).",
        )
        model: str = SchemaField(
            description="Model to request via the gateway (e.g. `gpt-4o`).",
            default="gpt-4o",
        )
        messages: List[Dict[str, Any]] = SchemaField(
            description="OpenAI-style messages array sent through the firewall.",
            default=[],
        )

    class Output(BlockSchemaOutput):
        completion: Dict[str, Any] = SchemaField(
            description=(
                "Raw JSON completion returned from the AgentShield/Vigil gateway "
                "after policy enforcement."
            ),
        )

    def __init__(self) -> None:
        super().__init__(
            # Keep this UUID stable once merged
            id="375e0fe5-e3cf-45c3-8262-e3deb46116df",
            input_schema=AgentShieldFirewallBlock.Input,
            output_schema=AgentShieldFirewallBlock.Output,
            # Sample data for the block test harness
            test_input={
                "gateway_url": "https://gateway.example.com/v1/chat/completions",
                "api_key": "ask_test_api_key",
                "tenant_id": "demo-enterprise",
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": "Say hello from AgentShield/Vigil test block.",
                    },
                ],
            },
            # For non-deterministic outputs we assert type (dict) instead of exact value
            test_output=("completion", dict),
            # Test mock: replaces call_gateway during automated block tests
            test_mock={
                "call_gateway": lambda gateway_url, api_key, tenant_id, payload: {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Hello from Vigil test stub.",
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 5,
                        "total_tokens": 12,
                    },
                    "model": payload.get("model", "gpt-4o"),
                }
            },
        )

    def call_gateway(
        self,
        gateway_url: str,
        api_key: str,
        tenant_id: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Actual HTTP call to the AgentShield gateway.

        Kept separate so the block test harness can override via `test_mock`
        without making real network calls.
        """
        response = requests.post(
            gateway_url,
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-Tenant-ID": tenant_id,
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def run(self, input_data: Input, **kwargs: Any) -> BlockOutput:  # type: ignore[reportReturnType]
        """
        Build a /v1/chat/completions payload, send it through the firewall,
        and yield the full JSON as `completion`.
        """
        payload: Dict[str, Any] = {
            "model": input_data.model,
            "messages": input_data.messages,
        }

        data = self.call_gateway(
            gateway_url=input_data.gateway_url,
            api_key=input_data.api_key,
            tenant_id=input_data.tenant_id,
            payload=payload,
        )

        yield "completion", data
