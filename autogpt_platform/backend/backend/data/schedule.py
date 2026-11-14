from typing import Annotated

from pydantic import StringConstraints, TypeAdapter

# Room for the editor's default name, "Run " and the workflow's name.
MAX_SCHEDULE_NAME_LENGTH = 1_000

ScheduleName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_SCHEDULE_NAME_LENGTH,
        pattern=r"\S",
    ),
]
_schedule_name = TypeAdapter(ScheduleName | None)


def normalize_schedule_name(name: str | None) -> str | None:
    return _schedule_name.validate_python(name)
