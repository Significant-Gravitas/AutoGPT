
from .app import app_router



from .user_middleware import UserIDMiddleware
from .user import user_router, afaas_user_router

from .agent_middleware import AgentMiddleware
from .agent import agent_router, afaas_agent_router
from .artifact import artifact_router, afaas_artifact_router

from dependencies.agents import get_agent