import feedparser

from backend.blocks.cajal_paper import CajalScientificPaperGeneratorBlock
from backend.blocks.cajal_paper_helpers import ArxivReference, reference_from_feed_entry


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
