
import uuid
from AFAAS.core.agents import PlannerAgent
from AFAAS.core.lib.sdk.logger import AFAASLogger

client_logger = AFAASLogger(__name__)

user_id: str = "A" + str(uuid.UUID("a1621e69-970a-4340-86e7-778d82e2137b"))
agent_settings: PlannerAgent.SystemSettings = PlannerAgent.SystemSettings(
    user_id=user_id
)

PLANNERAGENT = PlannerAgent.get_instance_from_settings(
            agent_settings=agent_settings,
            logger=client_logger,
        )