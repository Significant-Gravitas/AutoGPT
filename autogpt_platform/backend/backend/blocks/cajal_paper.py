import asyncio
from urllib.parse import urlencode

import feedparser
import ollama

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.cajal_paper_helpers import (
    ArxivReference,
    format_references,
    reference_from_feed_entry,
)
from backend.data.model import SchemaField
from backend.util.request import Requests, validate_url_host
from backend.util.settings import Settings

settings = Settings()

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_REQUEST_TIMEOUT_SECONDS = 30
OLLAMA_INFERENCE_TIMEOUT_SECONDS = 300
MAX_OLLAMA_CONTEXT_WINDOW = 16384
ALLOWED_OLLAMA_SCHEMES = ("http", "https")
DEFAULT_CAJAL_SYSTEM_PROMPT = (
    "You are CAJAL, a scientific paper authoring assistant. Write rigorous, "
    "citation-grounded academic drafts using only the verified references supplied "
    "by the workflow. Never invent citations, datasets, experimental results, "
    "authors, DOIs, or arXiv IDs."
)
PAPER_SECTIONS = (
    "Abstract",
    "Introduction",
    "Methodology",
    "Results",
    "Discussion",
    "Conclusion",
    "References",
)


class CajalScientificPaperGeneratorBlock(Block):
    class Input(BlockSchemaInput):
        topic: str = SchemaField(
            description="Research topic or question for the scientific paper.",
            placeholder="Federated learning for privacy-preserving medical imaging",
            min_length=3,
            max_length=500,
        )
        instructions: str = SchemaField(
            description="Additional paper requirements, constraints, or audience context.",
            default="",
            max_length=2000,
        )
        model: str = SchemaField(
            description=(
                "Ollama model tag to use for paper generation. Defaults to 'cajal' "
                "(Agnuxo/CAJAL-4B-P2PCLAW); pull or alias the model in your local "
                "Ollama instance before running this block. Any locally-available "
                "Ollama model tag is accepted."
            ),
            default="cajal",
        )
        ollama_host: str = SchemaField(
            description=(
                "Ollama host URL for local inference. Must include an http:// or "
                "https:// scheme."
            ),
            default="http://localhost:11434",
            advanced=True,
        )
        citation_count: int = SchemaField(
            description="Number of arXiv references to retrieve and require in the paper.",
            default=8,
            ge=1,
            le=12,
            advanced=True,
        )
        temperature: float = SchemaField(
            description="Ollama sampling temperature.",
            default=0.7,
            ge=0,
            le=2,
            advanced=True,
        )
        max_tokens: int = SchemaField(
            description="Maximum number of tokens Ollama should generate.",
            default=4096,
            ge=512,
            le=32768,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        paper: str = SchemaField(
            description="Generated scientific paper in Markdown format."
        )
        references: list[ArxivReference] = SchemaField(
            description="Verified arXiv references supplied to the generator."
        )
        citations_context: str = SchemaField(
            description="Formatted citation context sent to the local model."
        )
        prompt: list[dict[str, str]] = SchemaField(
            description="Messages sent to the local Ollama model."
        )

    def __init__(self):
        super().__init__(
            id="6d2b61a7-781f-4354-9388-cb84e085fc5d",
            description="Generates a citation-grounded scientific paper draft using verified arXiv references and a local Ollama model.",
            categories={BlockCategory.AI, BlockCategory.SEARCH, BlockCategory.TEXT},
            input_schema=CajalScientificPaperGeneratorBlock.Input,
            output_schema=CajalScientificPaperGeneratorBlock.Output,
            test_input={
                "topic": "Federated learning for privacy-preserving medical imaging",
                "instructions": "Emphasize reproducibility.",
                "citation_count": 2,
                "model": "cajal",
            },
            test_output=[
                ("paper", lambda paper: "## Abstract" in paper and "[1]" in paper),
                ("references", lambda references: len(references) == 2),
                (
                    "citations_context",
                    lambda context: "[1] Federated Learning Survey" in context,
                ),
                ("prompt", list),
            ],
            test_mock={
                "_fetch_arxiv_references": lambda *args: [
                    ArxivReference(
                        title="Federated Learning Survey",
                        authors=["A. Researcher", "B. Scientist"],
                        published="2024-01-01T00:00:00Z",
                        url="https://arxiv.org/abs/2401.00001",
                        pdf_url="https://arxiv.org/pdf/2401.00001",
                        summary="A survey of federated learning methods.",
                        categories=["cs.LG"],
                    ),
                    ArxivReference(
                        title="Privacy Preserving Medical Imaging",
                        authors=["C. Clinician"],
                        published="2023-05-01T00:00:00Z",
                        url="https://arxiv.org/abs/2305.00001",
                        pdf_url="https://arxiv.org/pdf/2305.00001",
                        summary="A paper on privacy in medical imaging.",
                        categories=["cs.CV"],
                    ),
                ],
                "_generate_paper": lambda *args: "# Draft\n\n## Abstract\nCited [1].",
            },
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        references = await self._fetch_arxiv_references(
            input_data.topic,
            input_data.citation_count,
        )
        if not references:
            yield "error", f"No arXiv references found for topic: {input_data.topic}"
            return

        citations_context = format_references(references)
        prompt = self._build_prompt(input_data, citations_context, len(references))
        try:
            paper = await self._generate_paper(input_data, prompt)
        except (ValueError, asyncio.TimeoutError) as error:
            yield "error", str(error)
            return

        yield "paper", paper
        yield "references", references
        yield "citations_context", citations_context
        yield "prompt", prompt

    async def _fetch_arxiv_references(
        self,
        topic: str,
        citation_count: int,
    ) -> list[ArxivReference]:
        query = urlencode(
            {
                "search_query": f"all:{topic}",
                "start": 0,
                "max_results": citation_count,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
        )
        response = await Requests(
            raise_for_status=True,
            retry_max_attempts=2,
        ).get(f"{ARXIV_API_URL}?{query}", timeout=ARXIV_REQUEST_TIMEOUT_SECONDS)
        feed = await asyncio.to_thread(feedparser.parse, response.content)
        entries = feed.get("entries") or []
        return [reference_from_feed_entry(entry) for entry in entries[:citation_count]]

    async def _generate_paper(
        self,
        input_data: Input,
        prompt: list[dict[str, str]],
    ) -> str:
        if not input_data.ollama_host.lower().startswith(
            tuple(f"{scheme}://" for scheme in ALLOWED_OLLAMA_SCHEMES)
        ):
            raise ValueError("ollama_host must include an http:// or https:// scheme.")
        parsed_host, _, _ = await validate_url_host(
            input_data.ollama_host,
            trusted_hostnames=[settings.config.ollama_host],
        )
        num_ctx = min(MAX_OLLAMA_CONTEXT_WINDOW, max(4096, input_data.max_tokens * 2))
        client = ollama.AsyncClient(host=parsed_host.geturl())
        response = await asyncio.wait_for(
            client.chat(
                model=input_data.model,
                messages=prompt,
                stream=False,
                options={
                    "temperature": input_data.temperature,
                    "num_predict": input_data.max_tokens,
                    "num_ctx": num_ctx,
                },
            ),
            timeout=OLLAMA_INFERENCE_TIMEOUT_SECONDS,
        )
        paper = response["message"]["content"].strip()
        if not paper:
            raise ValueError("Ollama returned an empty paper.")
        return paper

    def _build_prompt(
        self,
        input_data: Input,
        citations_context: str,
        reference_count: int,
    ) -> list[dict[str, str]]:
        sections = "\n".join(f"## {section}" for section in PAPER_SECTIONS)
        instructions = input_data.instructions.strip() or "No additional instructions."
        user_prompt = f"""
Research topic:
{input_data.topic}

Additional instructions:
{instructions}

Verified arXiv references:
{citations_context}

Write a complete Markdown scientific paper draft. Start with a single H1 title,
then use exactly these H2 sections and no extra H2 sections:
{sections}

Requirements:
- Cite only the supplied references with bracket citations like [1] and [2].
- Use at least {reference_count} supplied references.
- Do not invent references, DOIs, authors, arXiv IDs, datasets, or measured results.
- If empirical data is not supplied, make the Results section a proposed or expected
  results section and state that limitation clearly.
- The References section must list the supplied arXiv references that were cited.
""".strip()
        return [
            {"role": "system", "content": DEFAULT_CAJAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
