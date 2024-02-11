from __future__ import annotations

import enum
import time
import uuid
from typing import Optional

from pydantic import Field

from AFAAS.configs.schema import AFAASMessageType, AFAASModel
from AFAAS.lib.message_common import AFAASMessage


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
        return "QUE" + str(uuid.uuid4())

    message: str
    type: Optional[QuestionTypes] = None
    state: Optional[QuestionStates] = None
    items: Optional[list[QuestionItems]] = None


class Emiter(enum.Enum):
    USER = "USER"
    AGENT = "AGENT"


class MessageUserAgent(AFAASMessage):
    message_id: str = Field(default_factory=lambda: MessageUserAgent.generate_uuid())
    message_type: str = AFAASMessageType.AGENT_USER.value
    emitter: Emiter
    user_id: str
    agent_id: str  # Always PlannerAgent not ProxyAgent
    message: str
    question: Questions = None
    hidden: bool = False

    _table_name = "message_agent_user"

    def get_table_name(self):
        return self._table_name

    @classmethod
    def generate_uuid(cls):
        return "MAU" + str(time.time()) + str(uuid.uuid4())
