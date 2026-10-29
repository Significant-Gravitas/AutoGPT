"""What a block would create or change in a user's connected account, said in
the words the user will see in that service, for an approval card to show
before the block runs."""

from pydantic import BaseModel


class PreviewField(BaseModel):
    name: str
    # The field's type as the user would say it ("Text", "Money"), never the
    # API's own name for it.
    kind: str = ""
    # Its note and any setting the kind doesn't say ("precision 2"): the
    # approval binds them too, so the card must show them.
    detail: str = ""


class PreviewTable(BaseModel):
    name: str
    description: str = ""
    fields: list[PreviewField] = []


class ChangePreview(BaseModel):
    # The card's headline: what happens, then what it happens to.
    ask: str
    object: str | None = None
    tables: list[PreviewTable] = []
    # The block inputs the preview already shows, which the card leaves out
    # of its list of arguments.
    covers: list[str] = []
