"""What a collection holds, the limits on it, and the errors a write can meet."""

from datetime import date, datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from backend.util.exceptions import NotFoundError

MAX_COLLECTIONS_PER_USER = 50
MAX_ROWS = 5_000
MAX_FIELDS = 30
MAX_ROW_BYTES = 8_192
MAX_TEXT_CHARS = 2_000
MAX_OPTIONS = 20
MAX_NAME_CHARS = 80
MAX_FIELD_NAME_CHARS = 60
MAX_CONDITIONS = 3
MAX_QUERY_LIMIT = 200

Option = Annotated[str, Field(min_length=1, max_length=100)]


class FieldType(str, Enum):
    TEXT = "text"
    NUMBER = "number"
    DATE = "date"
    SELECT = "select"
    MULTI_SELECT = "multi_select"
    CHECKBOX = "checkbox"
    URL = "url"
    EMAIL = "email"


class CollectionField(BaseModel):
    name: str
    type: FieldType
    options: list[str] = Field(default_factory=list)
    unit: str | None = None
    # Earlier names of a renamed field, so a writer using one still lands here.
    aliases: list[str] = Field(default_factory=list)


class FieldSpec(BaseModel):
    """A field a writer declares up front."""

    name: str = Field(min_length=1, max_length=MAX_FIELD_NAME_CHARS)
    type: FieldType
    options: list[Option] = Field(default_factory=list, max_length=MAX_OPTIONS)
    unit: str | None = Field(default=None, max_length=12)


Writer = str
OWNER: Writer = "user"
OTTO: Writer = "otto"


def workflow_writer(graph_id: str) -> Writer:
    return f"workflow:{graph_id}"


def expert_writer(expert_id: str) -> Writer:
    return f"expert:{expert_id}"


class CollectionInfo(BaseModel):
    id: str
    name: str
    fields: list[CollectionField]
    key_field: str | None = None
    revision: int
    row_count: int = 0
    updated_at: datetime


class CollectionRowRecord(BaseModel):
    id: str
    key: str | None = None
    values: dict[str, Any]
    # Newer automated values held back over a person's edit, by field.
    pending: dict[str, Any] = Field(default_factory=dict)
    manual_fields: list[str] = Field(default_factory=list)
    written_by: Writer
    revision: int
    created_at: datetime
    updated_at: datetime


ConditionOp = Literal[
    "is",
    "is_not",
    "contains",
    "gt",
    "lt",
    "gte",
    "lte",
    "empty",
    "not_empty",
    # Dates against the reader's own day; last_days and next_days take a count.
    "today",
    "last_days",
    "next_days",
    "this_month",
]
Scalar = Annotated[str, Field(max_length=500)] | float | bool | None


class Condition(BaseModel):
    field: str = Field(min_length=1, max_length=MAX_FIELD_NAME_CHARS)
    op: ConditionOp
    value: Scalar = None


class Sort(BaseModel):
    field: str = Field(min_length=1, max_length=MAX_FIELD_NAME_CHARS)
    direction: Literal["asc", "desc"] = "asc"


class RowQuery(BaseModel):
    conditions: list[Condition] = Field(default_factory=list, max_length=MAX_CONDITIONS)
    sort: Sort | None = None
    search: str | None = Field(default=None, max_length=200)
    # Field -> options, from the chips a person flips; ORed within a field.
    chips: dict[str, list[Option]] = Field(default_factory=dict)
    limit: int = Field(default=50, ge=1, le=MAX_QUERY_LIMIT)
    offset: int = Field(default=0, ge=0, le=MAX_ROWS)
    today: date | None = Field(
        default=None, description="The reader's date; UTC's when not given."
    )


class RowPage(BaseModel):
    rows: list[CollectionRowRecord]
    total: int


AggregateOp = Literal["count", "sum", "average", "min", "max"]


class Aggregate(BaseModel):
    op: AggregateOp
    field: str | None = Field(default=None, max_length=MAX_FIELD_NAME_CHARS)


Bucket = Literal["day", "week", "month"]


class GroupValue(BaseModel):
    label: str | None
    value: float | None


class WriteResult(BaseModel):
    collection: CollectionInfo
    rows: list[CollectionRowRecord]
    added: int = 0
    updated: int = 0
    # Field -> how many cells kept a person's value over a newer automated one.
    held: dict[str, int] = Field(default_factory=dict)


class CollectionError(ValueError):
    """A write or a query the collection refuses; the message says why."""


class CollectionNotFoundError(NotFoundError):
    pass


class RowNotFoundError(NotFoundError):
    pass


class StaleRowError(CollectionError):
    """A person's edit was based on an older version of the row."""

    def __init__(self, message: str, current: CollectionRowRecord | None = None):
        super().__init__(message)
        self.current = current
