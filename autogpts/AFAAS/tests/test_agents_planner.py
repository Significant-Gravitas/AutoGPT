
import uuid
from autogpts.autogpt.autogpt.core.agents import PlannerAgent
from autogpts.AFAAS.app.sdk.forge_log import ForgeLogger

client_logger = ForgeLogger(__name__)

user_id: str = "A" + str(uuid.UUID("a1621e69-970a-4340-86e7-778d82e2137b"))
agent_settings: PlannerAgent.SystemSettings = PlannerAgent.SystemSettings(
    user_id=user_id
)

PLANNERAGENT = PlannerAgent.get_instance_from_settings(
            agent_settings=agent_settings,
            logger=client_logger,
        )