"""Turn the content-option models into Exa API payload fragments.

The same fragment is the ``contents`` object on ``/search`` and
``/findSimilar`` and the top level of a ``/contents`` request. Deprecated
options are translated to their documented replacements so a default payload
never carries a field Exa warns about.
"""

from typing import Any

from .content_settings import (
    ContentSettings,
    ContextAdvanced,
    ContextDisabled,
    ContextEnabled,
    ExtrasSettings,
    HighlightSettings,
    LivecrawlTypes,
    SummarySettings,
    TextAdvanced,
    TextDisabled,
    TextEnabled,
)

# Exa: "numSentences ... Currently mapped to a character budget of about 1333
# characters per sentence."
CHARACTERS_PER_SENTENCE = 1333

# Exa's documented livecrawl -> maxAgeHours mapping. "fallback" equals Exa's
# default (omit maxAgeHours); "preferred" has no equivalent and passes through.
LIVECRAWL_TO_MAX_AGE_HOURS: dict[LivecrawlTypes, int] = {
    LivecrawlTypes.ALWAYS: 0,
    LivecrawlTypes.NEVER: -1,
}


def process_contents_settings(contents: ContentSettings | None) -> dict[str, Any]:
    """Process ContentSettings into the Exa ``contents`` payload format."""
    if not contents:
        return {}

    payload: dict[str, Any] = {}
    text_value = process_text_field(contents.text)
    if text_value is not None:
        payload["text"] = text_value
    if contents.highlights is not None:
        payload["highlights"] = process_highlights(contents.highlights)
    if contents.summary is not None:
        payload["summary"] = process_summary(contents.summary)
    payload.update(
        process_freshness(
            max_age_hours=contents.max_age_hours,
            livecrawl=contents.livecrawl,
            livecrawl_timeout=contents.livecrawl_timeout,
            snapshot_as_of=contents.snapshot_as_of,
        )
    )
    if contents.subpages is not None:
        payload["subpages"] = contents.subpages
    if contents.subpage_target is not None:
        payload["subpageTarget"] = contents.subpage_target
    if contents.extras is not None:
        payload["extras"] = process_extras(contents.extras)
    context_value = process_context_field(contents.context)
    if context_value is not None:
        payload["context"] = context_value
    return payload


def process_text_field(
    text: bool | TextEnabled | TextDisabled | TextAdvanced | None,
) -> bool | dict[str, Any] | None:
    """Process text field for API payload."""
    if text is None:
        return None
    if isinstance(text, bool):
        return text
    if isinstance(text, TextDisabled):
        return False
    if isinstance(text, TextEnabled):
        return True
    text_dict = process_text_advanced(text)
    return text_dict if text_dict else True


def process_text_advanced(text: TextAdvanced) -> dict[str, Any]:
    """Build the advanced ``text`` object, omitting unset options."""
    text_dict: dict[str, Any] = {}
    if text.max_characters:
        text_dict["maxCharacters"] = text.max_characters
    if text.include_html_tags:
        text_dict["includeHtmlTags"] = True
    if text.verbosity is not None:
        text_dict["verbosity"] = text.verbosity.value
    if text.include_sections:
        text_dict["includeSections"] = [s.value for s in text.include_sections]
    if text.exclude_sections:
        text_dict["excludeSections"] = [s.value for s in text.exclude_sections]
    return text_dict


def process_highlights(highlights: HighlightSettings) -> bool | dict[str, Any]:
    """Build the ``highlights`` value.

    ``highlightsPerUrl`` is ignored by Exa and never sent. ``numSentences`` is
    deprecated, so it is sent as Exa's documented character budget instead.
    With nothing to steer, ``true`` asks for Exa's adaptive default.
    """
    highlights_dict: dict[str, Any] = {}
    if highlights.query:
        highlights_dict["query"] = highlights.query
    max_characters = highlight_max_characters(highlights)
    if max_characters is not None:
        highlights_dict["maxCharacters"] = max_characters
    return highlights_dict or True


def highlight_max_characters(highlights: HighlightSettings) -> int | None:
    if highlights.max_characters is not None:
        return highlights.max_characters
    if highlights.num_sentences is not None:
        return highlights.num_sentences * CHARACTERS_PER_SENTENCE
    return None


def process_summary(summary: SummarySettings) -> dict[str, Any]:
    summary_dict: dict[str, Any] = {}
    if summary.query is not None:
        summary_dict["query"] = summary.query
    if summary.output_schema is not None:
        summary_dict["schema"] = summary.output_schema
    return summary_dict


def process_freshness(
    *,
    max_age_hours: int | None,
    livecrawl: LivecrawlTypes | None,
    livecrawl_timeout: int | None,
    snapshot_as_of: str | None,
) -> dict[str, Any]:
    """Resolve the freshness options into payload keys.

    ``maxAgeHours`` wins over the deprecated ``livecrawl`` (Exa rejects both
    together). A snapshot never fetches live pages and Exa rejects it alongside
    any live-fetch option, so the fetch timeout is dropped for it.
    """
    freshness: dict[str, Any] = {}
    if max_age_hours is not None:
        freshness["maxAgeHours"] = max_age_hours
    elif livecrawl is not None and livecrawl in LIVECRAWL_TO_MAX_AGE_HOURS:
        freshness["maxAgeHours"] = LIVECRAWL_TO_MAX_AGE_HOURS[livecrawl]
    elif livecrawl == LivecrawlTypes.PREFERRED:
        freshness["livecrawl"] = livecrawl.value

    if snapshot_as_of:
        if freshness:
            raise ValueError(
                "Snapshot As Of serves stored page versions only, so it cannot be "
                "combined with Max Age Hours or Livecrawl. Clear one of them."
            )
        return {"snapshotAsOf": snapshot_as_of}

    if livecrawl_timeout is not None:
        freshness["livecrawlTimeout"] = livecrawl_timeout
    return freshness


def process_extras(extras: ExtrasSettings) -> dict[str, int]:
    """Build ``extras``; zero counts mean "none requested" and are omitted."""
    counts = {
        "links": extras.links,
        "imageLinks": extras.image_links,
        "richImageLinks": extras.rich_image_links,
    }
    return {key: count for key, count in counts.items() if count > 0}


def process_context_field(
    context: bool | dict | ContextEnabled | ContextDisabled | ContextAdvanced | None,
) -> bool | dict[str, int] | None:
    """Process the deprecated context field for API payload."""
    if context is None:
        return None
    if isinstance(context, bool):
        return context if context else None
    if isinstance(context, dict) and "maxCharacters" in context:
        return {"maxCharacters": context["maxCharacters"]}
    if isinstance(context, ContextDisabled):
        return None
    if isinstance(context, ContextEnabled):
        return True
    if isinstance(context, ContextAdvanced):
        if context.max_characters:
            return {"maxCharacters": context.max_characters}
        return True
    return None
