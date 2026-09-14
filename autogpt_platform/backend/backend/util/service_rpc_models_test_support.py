"""Support module for the RPC type-resolution regression in service_test.

Deliberately uses ``from __future__ import annotations`` and defines its own
model so the exposed functions carry *string* annotations that only resolve
in this module's namespace — exactly the shape of the data modules the
Prisma-less executor reaches over RPC.
"""

from __future__ import annotations

from pydantic import BaseModel


class SampleRecord(BaseModel):
    id: str
    count: int = 0


async def echo_record(record: SampleRecord, bump: int = 1) -> SampleRecord | None:
    """Model-valued request, optional-model return."""
    if record.id == "missing":
        return None
    return SampleRecord(id=record.id, count=record.count + bump)


async def list_records(ids: list[str]) -> list[SampleRecord]:
    return [SampleRecord(id=value) for value in ids]
