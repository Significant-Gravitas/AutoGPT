import enum
import uuid
from typing import Optional

from pydantic import BaseModel, Field

from AFAAS.interfaces.configuration.schema  import (
    AFAASMessageType, AFAASModel)


class QuestionTypes(str, enum.Enum):
    SWIFT_OR_IBAN = "bank_account"
    CREDIT_CARD = "credit_card"
    POSTAL_ADDRESS = "postal_address"
    EMAIL = "email"
    COUNTRY = "country"
    CITY = "city"
    NUMBER = "number"
    DATE = "date"
    DATE_TIME = "datetime"
    # ENUM = "enum"
    BOOLEAN = "boolean"
    STRING = "string"
    SELECT_LIST = "select_list"
    SUGGESTION_LIST = "open_list"
    MULTIPLE_CHOICE_LIST = "multiple_choice_list"


class QuestionStates(str, enum.Enum):
    BLOCKER = "blocker"
    OPTIONAL = "optional"


class QuestionItems(dict):
    value: str
    label: str


class Questions(AFAASModel):
    question_id: str = Field(default_factory=lambda: Questions.generate_uuid())

    @staticmethod
    def generate_uuid():
        return "Q" + str(uuid.uuid4())

    message: str
    type: Optional[QuestionTypes]
    state: Optional[QuestionStates]
    items: Optional[list[QuestionItems]]


class emiter(enum.Enum):
    USER = "USER"
    AGENT = "AGENT"


class MessageAgentUser(AFAASModel):
    message_id: str = "MUA" + str(uuid.uuid4())
    message_type: str = AFAASMessageType.AGENT_USER.value
    emitter: emiter
    user_id: str
    agent_id: str  # Always PlannerAgent not ProxyAgent
    message: str
    question: Questions
