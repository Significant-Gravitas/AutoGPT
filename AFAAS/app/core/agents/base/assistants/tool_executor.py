from logging import Logger

from AFAAS.app.core.agents.base.features.agentmixin import \
    AgentMixin


class ToolExecutor(AgentMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        from AFAAS.app.sdk import forge_log

        forge_log.ForgeLogger(__name__).debug(
            "ToolExecutor : Has not been implemented yet"
        )
        forge_log.ForgeLogger(__name__).debug(
            "ToolExecutor : Will be part of a @tool wrapper redisign"
        )
