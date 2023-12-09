from __future__ import annotations

from AFAAS.core.agents.base.features.agentmixin import \
    AgentMixin


class ToolExecutor(AgentMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        from AFAAS.core.lib.sdk.logger import AFAASLogger

        AFAASLogger(__name__).trace(
            "ToolExecutor : Has not been implemented yet"
        )
        AFAASLogger(__name__).trace(
            "ToolExecutor : Will be part of a @tool wrapper redisign"
        )
