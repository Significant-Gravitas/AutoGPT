from pydantic import BaseModel

from backend.data.model import SchemaField


class ContentSettings(BaseModel):
    text: dict = SchemaField(
        description="Text content settings",
        default={"maxCharacters": 1000, "includeHtmlTags": False},
    )
    highlights: dict = SchemaField(
        description="Highlight settings",
        default={"numSentences": 3, "highlightsPerUrl": 3},
    )
    summary: dict = SchemaField(
        description="Summary settings",
        default={"query": ""},
    )