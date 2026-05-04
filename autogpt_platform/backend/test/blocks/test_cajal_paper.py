import feedparser

from backend.blocks.cajal_paper import CajalScientificPaperGeneratorBlock
from backend.blocks.cajal_paper_helpers import (
    MAX_REFERENCE_ABSTRACT_CHARS,
    ArxivReference,
    clean_text,
    entry_authors,
    entry_pdf_url,
    format_reference,
    format_references,
    reference_from_feed_entry,
)


def test_reference_from_arxiv_feed_entry_parses_metadata():
    feed = feedparser.parse(
        """
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>https://arxiv.org/abs/2401.00001</id>
            <title>
              Federated Learning Survey
            </title>
            <summary>
              A survey of federated learning methods.
            </summary>
            <published>2024-01-01T00:00:00Z</published>
            <author><name>A. Researcher</name></author>
            <author><name>B. Scientist</name></author>
            <category term="cs.LG" />
            <link href="https://arxiv.org/abs/2401.00001" />
            <link href="https://arxiv.org/pdf/2401.00001" type="application/pdf" />
          </entry>
        </feed>
        """
    )
    reference = reference_from_feed_entry(feed["entries"][0])

    assert reference == ArxivReference(
        title="Federated Learning Survey",
        authors=["A. Researcher", "B. Scientist"],
        published="2024-01-01T00:00:00Z",
        url="https://arxiv.org/abs/2401.00001",
        pdf_url="https://arxiv.org/pdf/2401.00001",
        summary="A survey of federated learning methods.",
        categories=["cs.LG"],
    )


async def test_run_builds_paper_from_verified_references(monkeypatch):
    block = CajalScientificPaperGeneratorBlock()
    references = [
        ArxivReference(
            title="Federated Learning Survey",
            authors=["A. Researcher"],
            published="2024-01-01T00:00:00Z",
            url="https://arxiv.org/abs/2401.00001",
            pdf_url="https://arxiv.org/pdf/2401.00001",
            summary="A survey of federated learning methods.",
            categories=["cs.LG"],
        )
    ]
    captured_prompt = []

    async def fetch_references(topic: str, citation_count: int):
        assert topic == "Federated learning for medical imaging"
        assert citation_count == 1
        return references

    async def generate_paper(input_data, prompt):
        captured_prompt.extend(prompt)
        return "# Draft\n\n## Abstract\nThis draft cites prior work [1]."

    monkeypatch.setattr(block, "_fetch_arxiv_references", fetch_references)
    monkeypatch.setattr(block, "_generate_paper", generate_paper)

    outputs = {}
    async for output_name, output_data in block.run(
        block.Input(
            topic="Federated learning for medical imaging",
            citation_count=1,
        )
    ):
        outputs[output_name] = output_data

    assert outputs["paper"].startswith("# Draft")
    assert outputs["references"] == references
    assert "[1] Federated Learning Survey" in outputs["citations_context"]
    assert "Do not invent references" in captured_prompt[1]["content"]
    assert "## References" in captured_prompt[1]["content"]


async def test_run_yields_error_when_no_references(monkeypatch):
    block = CajalScientificPaperGeneratorBlock()

    async def fetch_no_references(topic: str, citation_count: int):
        return []

    async def generate_should_not_run(*args, **kwargs):
        raise AssertionError("_generate_paper must not be called when refs are empty")

    monkeypatch.setattr(block, "_fetch_arxiv_references", fetch_no_references)
    monkeypatch.setattr(block, "_generate_paper", generate_should_not_run)

    outputs = []
    async for output_name, output_data in block.run(
        block.Input(topic="No-results topic", citation_count=1)
    ):
        outputs.append((output_name, output_data))

    assert len(outputs) == 1
    name, message = outputs[0]
    assert name == "error"
    assert "No-results topic" in message


def test_build_prompt_pins_sections_and_uses_default_instructions():
    block = CajalScientificPaperGeneratorBlock()
    references = [
        ArxivReference(
            title="Sample",
            authors=["A. One"],
            published="2024-01-01T00:00:00Z",
            url="https://arxiv.org/abs/2401.00001",
            pdf_url="https://arxiv.org/pdf/2401.00001",
            summary="Summary.",
            categories=["cs.LG"],
        )
    ]
    citations_context = format_references(references)
    prompt = block._build_prompt(
        block.Input(topic="Topic X", instructions="   ", citation_count=1),
        citations_context,
        len(references),
    )

    assert prompt[0]["role"] == "system"
    assert prompt[1]["role"] == "user"
    user_content = prompt[1]["content"]
    assert "No additional instructions." in user_content
    for section in (
        "## Abstract",
        "## Introduction",
        "## Methodology",
        "## Results",
        "## Discussion",
        "## Conclusion",
        "## References",
    ):
        assert section in user_content
    assert "Use at least 1 supplied references." in user_content
    assert "[1] Sample" in user_content


def test_format_reference_handles_missing_authors_and_truncates_summary():
    long_summary = "x" * (MAX_REFERENCE_ABSTRACT_CHARS + 50)
    reference = ArxivReference(
        title="Edge Case Paper",
        authors=[],
        published="2024-01-01T00:00:00Z",
        url="https://arxiv.org/abs/2401.99999",
        pdf_url="https://arxiv.org/pdf/2401.99999",
        summary=long_summary,
        categories=[],
    )

    rendered = format_reference(7, reference)

    assert rendered.startswith("[7] Edge Case Paper")
    assert "Authors: Unknown authors" in rendered
    assert "Categories: uncategorized" in rendered
    truncated = "x" * MAX_REFERENCE_ABSTRACT_CHARS
    assert f"Abstract: {truncated}" in rendered
    assert long_summary not in rendered


def test_entry_authors_falls_back_to_singular_author_field():
    feed = feedparser.parse(
        """
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>https://arxiv.org/abs/2401.00002</id>
            <title>Solo Author Paper</title>
            <author>Lone Researcher</author>
          </entry>
        </feed>
        """
    )
    entry = feed["entries"][0]

    assert entry_authors(entry) == ["Lone Researcher"]


def test_entry_authors_returns_empty_when_no_author_metadata():
    feed = feedparser.parse(
        """
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>https://arxiv.org/abs/2401.00003</id>
            <title>Anonymous Paper</title>
          </entry>
        </feed>
        """
    )

    assert entry_authors(feed["entries"][0]) == []


def test_entry_pdf_url_falls_back_to_entry_link_when_no_pdf_link():
    feed = feedparser.parse(
        """
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>https://arxiv.org/abs/2401.00004</id>
            <title>HTML Only Paper</title>
            <link href="https://arxiv.org/abs/2401.00004" />
          </entry>
        </feed>
        """
    )

    assert (
        entry_pdf_url(feed["entries"][0])
        == "https://arxiv.org/abs/2401.00004"
    )


def test_clean_text_collapses_whitespace_and_strips():
    assert clean_text("  hello\n\tworld  ") == "hello world"
    assert clean_text("") == ""
