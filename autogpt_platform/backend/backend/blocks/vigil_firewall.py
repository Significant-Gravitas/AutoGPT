from typing import Any, Dict, List

import requests
from backend.data.block import Block, BlockSchemaInput, BlockSchemaOutput, BlockOutput


class AgentShieldFirewallBlock(Block):
    """
    Block: AgentShield Firewall + Analytics

    This block routes LLM calls through the AgentShield Gateway so that:
    - per-agent / per-tenant policies are enforced
    - PII / unsafe prompts can be blocked
    - every request/response is logged for analytics (Vigil)

    Inputs:
        gateway_url: full URL to AgentShield gateway endpoint
                     e.g. https://gateway.agentshield.yourdomain.com/v1/chat/completions
        api_key:     AgentShield API key for this agent
        tenant_id:   Tenant identifier (e.g. "demo-enterprise")
        model:       Model name to request from AgentShield (e.g. "gpt-4o")
        messages:    Standard OpenAI-style messages array
    Outputs:
        completion:  Raw JSON response returned by the AgentShield gateway
    """

    class Input(BlockSchemaInput):
        gateway_url: str
        api_key: str
        tenant_id: str
        model: str
        messages: List[Dict[str, Any]]

    class Output(BlockSchemaOutput):
        completion: Dict[str, Any]

    def __init__(self) -> None:
        super().__init__(
            # Unique ID for the block (pre-generated UUID)
            id="375e0fe5-e3cf-45c3-8262-e3deb46116df",
            input_schema=AgentShieldFirewallBlock.Input,
            output_schema=AgentShieldFirewallBlock.Output,
            # Sample input for testing in the UI / test runner
            test_input={
                "gateway_url": "https://gateway.example.com/v1/chat/completions",
                "api_key": "ask_test_api_key",
                "tenant_id": "demo-enterprise",
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": "Say hello from AgentShield test block."}
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
                                "content": "Hello from AgentShield test stub."
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
        Actual HTTP call to AgentShield Gateway.

        This is kept as a separate method so the AutoGPT test harness can
        override it via `test_mock` without performing real network calls.
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

    def run(self, input_data: Input, **kwargs: Any) -> BlockOutput:
        """
        Execute the block:

        1. Build a standard /v1/chat/completions payload from inputs
        2. Send it through the AgentShield gateway
        3. Yield the full completion JSON as `completion`
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

        # Yield a single output field named "completion"
        yield "completion", data
