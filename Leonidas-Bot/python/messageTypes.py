from enum import Enum

class MessageTypes(Enum):

    DIRECT_MESSAGE = "direct_message"

    CHANNEL_MENTION = "channel_mention"

    GO_PRIVATE = "go_private"

    CHAT_RESET = "chat_reset"

    NO_RESPONSE = "no_response"

    RESPOND = "general_response"