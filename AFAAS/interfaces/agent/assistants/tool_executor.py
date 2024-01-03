from __future__ import annotations

from AFAAS.interfaces.agent.features.agentmixin import AgentMixin
from AFAAS.lib.sdk.logger import AFAASLogger

LOG=AFAASLogger(name="autogpt")

class ToolExecutor(AgentMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        from AFAAS.lib.sdk.logger import AFAASLogger

        LOG.trace(
            "ToolExecutor : Has not been implemented yet"
        )
        LOG.trace(
            "ToolExecutor : Will be part of a @tool wrapper redisign"
        )
