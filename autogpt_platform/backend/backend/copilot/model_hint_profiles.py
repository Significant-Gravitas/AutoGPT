"""The specialists a model hint can name, and the hidden block that names one.

Each profile is a block AutoPilot can already run with ``run_capability``, the
model to pick in it, and what it does better than Claude, worded as our claim
rather than a benchmark. ``model_hints.py`` asks Jev which one fits a request.
"""

from pydantic import BaseModel

MODEL_HINT_TAG = "model_hint"

_PERPLEXITY_BLOCK = "c8a5f2e9-8b3d-4a7e-9f6c-1d5e3c9b7a4f"
_TEXT_GENERATOR_BLOCK = "1f292d4a-41a4-4977-9684-7c8d560b9f91"


class ModelProfile(BaseModel):
    key: str
    block_id: str
    model: str
    # What Jev matches a request against.
    when: str
    # What AutoPilot is told the specialist does better.
    strength: str
    # The block's input fields besides ``model``.
    inputs: str


PROFILES: tuple[ModelProfile, ...] = (
    ModelProfile(
        key="web_research",
        block_id=_PERPLEXITY_BLOCK,
        model="perplexity/sonar-pro",
        when="the request needs current facts from the web with sources, such as"
        " recent news, prices, statistics, or details about a company or product",
        strength="It searches the live web and answers with citations in one call,"
        " where you would need several rounds of web_search and reading.",
        inputs='"prompt": the question, with any context it needs',
    ),
    ModelProfile(
        key="deep_research",
        block_id=_PERPLEXITY_BLOCK,
        model="perplexity/sonar-deep-research",
        when="the request asks for a thorough research report across many sources,"
        " such as a market, competitor or literature review",
        strength="It runs a long multi-source investigation and writes a cited"
        " report. It is slow and costs more, so use it only for a real report.",
        inputs='"prompt": the research brief',
    ),
    ModelProfile(
        key="image",
        block_id="ed1ae7a0-b770-4089-b520-1f0005fad19a",
        model="Nano Banana Pro",
        when="the request asks to create a new image, picture, logo, illustration"
        " or photo, not a diagram, chart or document",
        strength="It generates images, which you cannot do yourself.",
        inputs='"prompt": a detailed description of the image; optional "size"'
        ' and "style"',
    ),
    ModelProfile(
        key="video",
        block_id="530cf046-2ce0-4854-ae2c-659db17c7a46",
        model="fal-ai/veo3",
        when="the request asks to create a short video clip from a description",
        strength="It generates short video clips, which you cannot do yourself.",
        inputs='"prompt": a detailed description of the clip',
    ),
    ModelProfile(
        key="translation",
        block_id=_TEXT_GENERATOR_BLOCK,
        model="cohere/command-a-translate-08-2025",
        when="the main task is translating a text of a paragraph or more, which"
        " the user gave, into another language",
        strength="It is a model built for translation; we expect more faithful,"
        " natural translations of longer business text than your own.",
        inputs='"prompt": the instruction and the full text to translate',
    ),
    ModelProfile(
        key="long_document",
        block_id=_TEXT_GENERATOR_BLOCK,
        model="google/gemini-2.5-pro",
        when="the request is to read, summarise or answer questions about a very"
        " long text or many documents at once, hundreds of pages or more",
        strength="It reads about a million tokens in one call, five times your"
        " window, so the whole text fits at once.",
        inputs='"prompt": the question and the full text, or a file reference',
    ),
)

PROFILES_BY_KEY = {profile.key: profile for profile in PROFILES}


def render_hint(profile: ModelProfile) -> str:
    """The ``<model_hint>`` block, with the separator a leading block carries."""
    return (
        f"<{MODEL_HINT_TAG}>\n"
        "A routing check matched this request to a specialist model. "
        f"{profile.strength} To use it, call run_capability with "
        f'id="block:{profile.block_id}" and input {{"model": "{profile.model}", '
        f"{profile.inputs}}}, then build your reply on its result. This is a "
        "suggestion: ignore it if the request needs something else.\n"
        f"</{MODEL_HINT_TAG}>\n\n"
    )
