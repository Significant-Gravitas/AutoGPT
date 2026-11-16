"""The relay's reader of what the person has said, for the ending screens.

A call whose ending shows the person's profile card filling in reads it after
each of their turns, from the captions so far (live); the post-call job reads
it once more from the transcript of record for the screens that review the
call afterwards. Both go through ``onboarding_interview_details``, so the
prompt, schema and quote check are the REST API's too.
"""

from typing import Protocol, runtime_checkable

from google.genai import Client

from backend.copilot.onboarding_interview_details import DetailsRead, extract_details
from backend.voice_relay.config import DETAILS_MODEL

# The arm whose card fills in while the person talks.
LIVE_DETAILS_ENDINGS = frozenset({"02-profile-card"})
# The arms that show the details once the call is over.
POSTCALL_DETAILS_ENDINGS = frozenset(
    {"02-profile-card", "03-summary-chips", "08-transcript-highlights"}
)


@runtime_checkable
class DetailsReader(Protocol):
    async def __call__(
        self, lines: list[tuple[str, str]], *, live: bool
    ) -> DetailsRead: ...


class GeminiDetailsReader:
    def __init__(self, client: Client, *, model: str = DETAILS_MODEL) -> None:
        self._client = client
        self._model = model

    async def __call__(
        self, lines: list[tuple[str, str]], *, live: bool
    ) -> DetailsRead:
        return await extract_details(
            self._client, model=self._model, lines=lines, live=live
        )
