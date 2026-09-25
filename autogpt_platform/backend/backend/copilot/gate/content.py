"""The content judge: does an outside read carry instructions addressed to an agent?

Same call path as the supervisor (``classifier.py``) with a second rubric, and
the same rule: every failure shape holds the read. The prompt matches
``scripts/supervisor_eval`` so what ships is what was measured.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from backend.copilot.config import ChatConfig
from backend.util.llm.providers import call_provider_openai_compat_sync

from .classifier import fence, parse_answer

logger = logging.getLogger(__name__)
config = ChatConfig()

CONTENT_RUBRIC = (Path(__file__).parent / "content_rubric.txt").read_text(
    encoding="utf-8"
)

_UNCHECKED = "this content could not be checked for instructions"
_NO_PASSAGE = "the judge flagged this content but quoted no passage"


class Image(BaseModel):
    model_config = ConfigDict(frozen=True)

    mime_type: str
    data_base64: str


class ContentVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    held: bool
    passage: str = ""
    judged: bool = True


async def judge_content(
    *, source: str, text: str, images: tuple[Image, ...] = ()
) -> ContentVerdict:
    """Anything but a well-formed "clean" holds; ``judged=False`` marks a failure."""
    # Deferred, and private: copilot.service imports the tool registry, whose
    # BaseTool imports this gate.
    from backend.copilot.service import _get_aux_client

    body = fence("FETCHED CONTENT", f"source: {source}\n\n{text}")
    content: str | list[dict[str, Any]] = body
    if images:
        content = [{"type": "text", "text": body}] + [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{i.mime_type};base64,{i.data_base64}"},
            }
            for i in images
        ]
    try:
        response = await asyncio.wait_for(
            call_provider_openai_compat_sync(
                client=_get_aux_client(),
                model=config.gate_content_model,
                messages=[
                    {"role": "system", "content": CONTENT_RUBRIC},
                    {"role": "user", "content": content},
                ],
                max_tokens=200,
                timeout_seconds=config.content_judge_timeout_s,
            ),
            timeout=config.content_judge_timeout_s + 1,
        )
        raw = (response.choices[0].message.content or "") if response.choices else ""
    except Exception:
        logger.warning(f"Content judge failed on {source[:80]}", exc_info=True)
        return ContentVerdict(held=True, passage=_UNCHECKED, judged=False)

    verdict = parse_answer(_normalised(raw), ("clean", "hold"), "passage")
    if verdict is None:
        logger.warning(f"Content judge returned an unusable body for {source[:80]}")
        return ContentVerdict(held=True, passage=_UNCHECKED, judged=False)
    if verdict[0] == "clean":
        return ContentVerdict(held=False)
    passage = verdict[1].strip().strip('"')
    if not passage or passage.lower() == "none":
        passage = _NO_PASSAGE
    return ContentVerdict(held=True, passage=passage)


def _normalised(raw: str) -> str:
    """Sonnet 5 often echoes the rubric's ``clean|hold`` line, or answers with
    the passage line alone; that line is then the finding."""
    text = raw.strip()
    first, _, rest = text.partition("\n")
    if first.strip().strip("*`").lower().replace(" ", "") in (
        "clean|hold",
        "hold|clean",
    ):
        text = rest.strip()
    if not text.lower().startswith("passage:"):
        return text
    found = text.split(":", 1)[1].strip().strip("\"'").lower()
    return "clean" if found == "none" else f"hold\n{text}"
