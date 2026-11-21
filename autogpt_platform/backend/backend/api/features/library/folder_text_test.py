"""A folder's name, icon and colour are checked where they are written, against
the folder request models' rules, so every caller is held to them."""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import prisma.models
import pytest

from backend.api.features.library import db as library_db
from backend.api.features.library.exceptions import FolderValidationError
from backend.util.text_limits import MAX_NAME_LENGTH

REFUSED = {
    "name too long": ("name", "n" * (MAX_NAME_LENGTH + 1)),
    "icon too long": ("icon", "i" * (MAX_NAME_LENGTH + 1)),
    "color too long": ("color", "#" + "a" * MAX_NAME_LENGTH),
    "color not #RRGGBB": ("color", "red"),
}


class _Reached(Exception):
    """The write got as far as its query."""


@pytest.mark.parametrize("writer", ["create", "update"])
@pytest.mark.parametrize("field, value", REFUSED.values(), ids=REFUSED.keys())
async def test_text_the_request_models_refuse_is_refused_before_any_query(
    writer: str, field: str, value: str
):
    queried = AssertionError("queried")
    with _queries(write=queried, ownership=queried):
        with pytest.raises(FolderValidationError, match=field) as refused:
            await _write(writer, **{"name": "Folder", field: value})

    assert value not in str(refused.value)


@pytest.mark.parametrize("writer", ["create", "update"])
async def test_text_at_the_bounds_is_written(writer: str):
    with _queries(write=_Reached()), pytest.raises(_Reached):
        await _write(
            writer,
            name="n" * MAX_NAME_LENGTH,
            icon="i" * MAX_NAME_LENGTH,
            color="#A1b2C3",
        )


async def _write(writer: str, **text: str):
    if writer == "create":
        return await library_db.create_folder(user_id="user-1", **text)
    return await library_db.update_folder(folder_id="f-1", user_id="user-1", **text)


@contextmanager
def _queries(write: Exception, ownership: Exception | None = None):
    """The folder write raises *write*; the update's ownership read raises
    *ownership*, or answers as if the folder were the user's."""
    client = MagicMock(
        create=AsyncMock(side_effect=write), update=AsyncMock(side_effect=write)
    )
    with (
        patch.object(prisma.models.LibraryFolder, "prisma", return_value=client),
        patch.object(library_db, "get_folder", AsyncMock(side_effect=ownership)),
    ):
        yield
