import uuid

from autogpt.core.runner.cli_web_app.messaging.simple import Role, SimpleMessageBroker
from autogpt.core.runner.cli_web_app.server.agent import agent_context
from autogpt.core.runner.cli_web_app.server.factory import agent_factory_context


def get_message_broker() -> SimpleMessageBroker:
    message_channel_name = "autogpt"
    message_broker = SimpleMessageBroker()
    message_broker.create_message_channel(message_channel_name)

    message_broker.register_listener(
        message_channel="autogpt",
        listener=lambda message: APPLICATION_MESSAGE_QUEUE.put("autogpt", message),
        message_filter=MessageFilters.is_server_message,
    )

    message_broker.register_listener(
        message_channel="autogpt",
        listener=agent_factory_context.parse_goals,
        message_filter=MessageFilters.is_parse_goals_message,
    )

    message_broker.register_listener(
        message_channel="autogpt",
        listener=agent_factory_context.bootstrap_agent,
        message_filter=MessageFilters.is_user_bootstrap_message,
    )

    message_broker.register_listener(
        message_channel="autogpt",
        listener=agent_context.launch_agent,
        message_filter=MessageFilters.is_user_launch_message,
    )

    return message_broker


MESSAGE_BROKER = get_message_broker()
APPLICATION_UUID = uuid.uuid4()
APPLICATION_EMITTER = MESSAGE_BROKER.get_emitter(
    channel_name="autogpt",
    sender_uuid=APPLICATION_UUID,
    sender_name="autogpt-server",
    sender_role=Role.APPLICATION_SERVER,
)
APPLICATION_USERS = {}
