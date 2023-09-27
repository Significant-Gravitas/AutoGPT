import enum
import uuid
from pydantic import BaseModel
from typing import Optional



class AgentUserResponse(BaseModel):
    pass


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
    ENUM = "enum"
    OPEN_LIST = "open_list"
    BOOLEAN = "boolean"
    STRING = "string"


class QuestionStates(str, enum.Enum):
    BLOCKER = "blocker"
    OPTIONAL = "optional"


class QuestionItems(dict):
    value: str
    label: str


class Questions(AgentUserResponse):
    id: str
    message: str
    type: Optional[QuestionTypes]
    state: Optional[QuestionStates]
    items: Optional[
        list[QuestionItems]
    ]  # labeled values of enum, boolean , open list // or value / label

    def generate_new_id() -> str:
        return "Q" + str(uuid.uuid4())