"""Typed views of the SpecStream API responses the blocks use."""

from datetime import datetime
from typing import Any

from pydantic import ValidationError

from backend.sdk import BaseModel, Field

TERMINAL_STATES = frozenset({"completed", "partial", "failed", "canceled", "expired"})


class UnsupportedClause(BaseModel):
    field_id: str = ""
    reason: str = ""


class ReadingLimit(BaseModel):
    max: int = 25
    inferred: bool = True


class ReadingAst(BaseModel):
    limit: ReadingLimit = Field(default_factory=ReadingLimit)


class Reading(BaseModel):
    """How SpecStream read a request written in plain words."""

    status: str
    ast_hash: str | None = None
    ast: ReadingAst | None = None
    canonical_sentence: str | None = None
    notice: str | None = None
    unsupported: list[UnsupportedClause] = Field(default_factory=list)

    @property
    def stated_count(self) -> int | None:
        """The row count the request itself asks for, or None when it names none."""
        if self.ast is None or self.ast.limit.inferred:
            return None
        return self.ast.limit.max

    @property
    def is_runnable(self) -> bool:
        return self.status == "runnable" and bool(self.ast_hash)


class ExistingExecution(BaseModel):
    find_id: str | None = None


class Quote(BaseModel):
    """A signed price for one reading. Nothing runs until it is authorized."""

    quote_id: str
    quote_hash: str
    currency: str = "USD"
    maximum_cents: int
    expires_at: datetime
    scope: dict[str, Any] = Field(default_factory=dict)
    terms: dict[str, Any] = Field(default_factory=dict)
    existing_execution: ExistingExecution | None = None


class Execution(BaseModel):
    reservation_id: str
    find_id: str | None = None
    maximum_cents: int
    state: str
    replay: bool = False


class Commerce(BaseModel):
    amount_authorized_cents: int = 0
    amount_captured_cents: int = 0
    delivered_rows: int = 0


class RunStatus(BaseModel):
    """A run, which the API calls a find."""

    find_id: str
    status: str
    requested_count: int = 0
    accepted_count: int = 0
    stop_reason: str | None = None
    queue_position: int | None = None
    commerce: Commerce | None = None

    @property
    def is_done(self) -> bool:
        return self.status in TERMINAL_STATES


class Provenance(BaseModel):
    source_url: str | None = None
    observed_at: datetime | None = None


class ApiCell(BaseModel):
    column_id: str
    label: str
    value: Any = None
    display: str | None = None
    provenance: Provenance | None = None


class ApiRow(BaseModel):
    entity_id: str
    ordinal: int = 0
    cells: list[ApiCell] = Field(default_factory=list)
    worded_evidence: dict[str, Any] | None = None


class RowsPage(BaseModel):
    find_id: str
    status: str
    accepted_count: int = 0
    rows: list[ApiRow] = Field(default_factory=list)
    next_cursor: str | None = None


class SourcedValue(BaseModel):
    """One value of a row with its source URL and read time."""

    column_id: str
    label: str
    value: Any = None
    display: str | None = None
    source_url: str | None = None
    read_at: str | None = None


class DatasetRow(BaseModel):
    entity_id: str
    position: int
    values: dict[str, Any]
    cells: list[SourcedValue]

    @classmethod
    def from_api(cls, row: ApiRow) -> "DatasetRow":
        witnesses = _witnesses(row)
        cells = []
        for cell in row.cells:
            source = _source(cell, witnesses)
            cells.append(
                SourcedValue(
                    column_id=cell.column_id,
                    label=cell.label,
                    value=cell.value,
                    display=cell.display,
                    source_url=source.source_url if source else None,
                    read_at=(
                        source.observed_at.isoformat()
                        if source and source.observed_at
                        else None
                    ),
                )
            )
        return cls(
            entity_id=row.entity_id,
            position=row.ordinal,
            values={cell.column_id: cell.value for cell in row.cells},
            cells=cells,
        )


DETAIL_PREFIX = "detail:"
MATCHES_COLUMN = "matches"


def _witness(raw: Any) -> Provenance | None:
    """The source URL and read time of a worded evidence witness, if it has them."""
    if not isinstance(raw, dict) or not raw.get("source_url"):
        return None
    try:
        return Provenance.model_validate(
            {"source_url": raw["source_url"], "observed_at": raw.get("observed_at")}
        )
    except ValidationError:
        return Provenance(source_url=str(raw["source_url"]))


def _witnesses(row: ApiRow) -> dict[str, Provenance]:
    """Sources of the facts read in words, by detail ID, from the row's evidence."""
    details = (row.worded_evidence or {}).get("details") or []
    found: dict[str, Provenance] = {}
    for item in details:
        if isinstance(item, dict) and (source := _witness(item.get("witness"))):
            found.setdefault(str(item.get("detail_id", "")), source)
    return found


def _source(cell: ApiCell, witnesses: dict[str, Provenance]) -> Provenance | None:
    """The cell's own source, with any part it lacks filled from its witness."""
    witnessed = _witnessed(cell.column_id, witnesses)
    if cell.provenance is None:
        return witnessed
    if witnessed is None:
        return cell.provenance
    return Provenance(
        source_url=cell.provenance.source_url or witnessed.source_url,
        observed_at=cell.provenance.observed_at or witnessed.observed_at,
    )


def _witnessed(column_id: str, witnesses: dict[str, Provenance]) -> Provenance | None:
    if column_id.startswith(DETAIL_PREFIX):
        return witnesses.get(column_id.removeprefix(DETAIL_PREFIX))
    if column_id == MATCHES_COLUMN:
        return next(iter(witnesses.values()), None)
    return None
