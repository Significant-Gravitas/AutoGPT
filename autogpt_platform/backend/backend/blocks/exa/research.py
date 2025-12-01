"""
Exa Research Task Blocks

Provides asynchronous research capabilities that explore the web, gather sources,
synthesize findings, and return structured results with citations.
"""

import asyncio
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    Requests,
    SchemaField,
)

from ._config import exa


class ResearchModel(str, Enum):
    """Available research models."""

    FAST = "exa-research-fast"
    STANDARD = "exa-research"
    PRO = "exa-research-pro"


class ResearchStatus(str, Enum):
    """Research task status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"


class ResearchCostModel(BaseModel):
    """Cost breakdown for a research request."""

    total: float
    num_searches: int
    num_pages: int
    reasoning_tokens: int

    @classmethod
    def from_api(cls, data: dict) -> "ResearchCostModel":
        """Convert API response, rounding fractional counts to integers."""
        return cls(
            total=data.get("total", 0.0),
            num_searches=int(round(data.get("numSearches", 0))),
            num_pages=int(round(data.get("numPages", 0))),
            reasoning_tokens=int(round(data.get("reasoningTokens", 0))),
        )


class ResearchOutputModel(BaseModel):
    """Research output with content and optional structured data."""

    content: str
    parsed: Optional[Dict[str, Any]] = None


class ResearchTaskModel(BaseModel):
    """Stable output model for research tasks."""

    research_id: str
    created_at: int
    model: str
    instructions: str
    status: str
    output_schema: Optional[Dict[str, Any]] = None
    output: Optional[ResearchOutputModel] = None
    cost_dollars: Optional[ResearchCostModel] = None
    finished_at: Optional[int] = None
    error: Optional[str] = None

    @classmethod
    def from_api(cls, data: dict) -> "ResearchTaskModel":
        """Convert API response to our stable model."""
        output_data = data.get("output")
        output = None
        if output_data:
            output = ResearchOutputModel(
                content=output_data.get("content", ""),
                parsed=output_data.get("parsed"),
            )

        cost_data = data.get("costDollars")
        cost = None
        if cost_data:
            cost = ResearchCostModel.from_api(cost_data)

        return cls(
            research_id=data.get("researchId", ""),
            created_at=data.get("createdAt", 0),
            model=data.get("model", "exa-research"),
            instructions=data.get("instructions", ""),
            status=data.get("status", "pending"),
            output_schema=data.get("outputSchema"),
            output=output,
            cost_dollars=cost,
            finished_at=data.get("finishedAt"),
            error=data.get("error"),
        )


class ExaCreateResearchBlock(Block):
    """Create an asynchronous research task that explores the web and synthesizes findings."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        instructions: str = SchemaField(
            description="Research instructions - clearly define what information to find, how to conduct research, and desired output format.",
            placeholder="Research the top 5 AI coding assistants, their features, pricing, and user reviews",
        )
        model: ResearchModel = SchemaField(
            default=ResearchModel.STANDARD,
            description="Research model: 'fast' for quick results, 'standard' for balanced quality, 'pro' for thorough analysis",
        )
        output_schema: Optional[dict] = SchemaField(
            default=None,
            description="JSON Schema to enforce structured output. When provided, results are validated and returned as parsed JSON.",
            advanced=True,
        )
        wait_for_completion: bool = SchemaField(
            default=True,
            description="Wait for research to complete before returning. Ensures you get results immediately.",
        )
        polling_timeout: int = SchemaField(
            default=600,
            description="Maximum time to wait for completion in seconds (only if wait_for_completion is True)",
            advanced=True,
            ge=1,
            le=3600,
        )

    class Output(BlockSchemaOutput):
        research_id: str = SchemaField(
            description="Unique identifier for tracking this research request"
        )
        status: str = SchemaField(description="Final status of the research")
        model: str = SchemaField(description="The research model used")
        instructions: str = SchemaField(
            description="The research instructions provided"
        )
        created_at: int = SchemaField(
            description="When the research was created (Unix timestamp in ms)"
        )
        output_content: Optional[str] = SchemaField(
            description="Research output as text (only if wait_for_completion was True and completed)"
        )
        output_parsed: Optional[dict] = SchemaField(
            description="Structured JSON output (only if wait_for_completion and outputSchema were provided)"
        )
        cost_total: Optional[float] = SchemaField(
            description="Total cost in USD (only if wait_for_completion was True and completed)"
        )
        elapsed_time: Optional[float] = SchemaField(
            description="Time taken to complete in seconds (only if wait_for_completion was True)"
        )

    def __init__(self):
        super().__init__(
            id="a1f2e3d4-c5b6-4a78-9012-3456789abcde",
            description="Create research task with optional waiting - explores web and synthesizes findings with citations",
            categories={BlockCategory.SEARCH, BlockCategory.AI},
            input_schema=ExaCreateResearchBlock.Input,
            output_schema=ExaCreateResearchBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = "https://api.exa.ai/research/v1"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        payload: Dict[str, Any] = {
            "model": input_data.model.value,
            "instructions": input_data.instructions,
        }

        if input_data.output_schema:
            payload["outputSchema"] = input_data.output_schema

        response = await Requests().post(url, headers=headers, json=payload)
        data = response.json()

        research_id = data.get("researchId", "")

        if input_data.wait_for_completion:
            start_time = time.time()
            get_url = f"https://api.exa.ai/research/v1/{research_id}"
            get_headers = {"x-api-key": credentials.api_key.get_secret_value()}
            check_interval = 10

            while time.time() - start_time < input_data.polling_timeout:
                poll_response = await Requests().get(url=get_url, headers=get_headers)
                poll_data = poll_response.json()

                status = poll_data.get("status", "")

                if status in ["completed", "failed", "canceled"]:
                    elapsed = time.time() - start_time
                    research = ResearchTaskModel.from_api(poll_data)

                    yield "research_id", research.research_id
                    yield "status", research.status
                    yield "model", research.model
                    yield "instructions", research.instructions
                    yield "created_at", research.created_at
                    yield "elapsed_time", elapsed

                    if research.output:
                        yield "output_content", research.output.content
                        yield "output_parsed", research.output.parsed

                    if research.cost_dollars:
                        yield "cost_total", research.cost_dollars.total
                    return

                await asyncio.sleep(check_interval)

            raise ValueError(
                f"Research did not complete within {input_data.polling_timeout} seconds"
            )
        else:
            yield "research_id", research_id
            yield "status", data.get("status", "pending")
            yield "model", data.get("model", input_data.model.value)
            yield "instructions", data.get("instructions", input_data.instructions)
            yield "created_at", data.get("createdAt", 0)


class ExaGetResearchBlock(Block):
    """Get the status and results of a research task."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        research_id: str = SchemaField(
            description="The ID of the research task to retrieve",
            placeholder="01jszdfs0052sg4jc552sg4jc5",
        )
        include_events: bool = SchemaField(
            default=False,
            description="Include detailed event log of research operations",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        research_id: str = SchemaField(description="The research task identifier")
        status: str = SchemaField(
            description="Current status: pending, running, completed, canceled, or failed"
        )
        instructions: str = SchemaField(
            description="The original research instructions"
        )
        model: str = SchemaField(description="The research model used")
        created_at: int = SchemaField(
            description="When research was created (Unix timestamp in ms)"
        )
        finished_at: Optional[int] = SchemaField(
            description="When research finished (Unix timestamp in ms, if completed/canceled/failed)"
        )
        output_content: Optional[str] = SchemaField(
            description="Research output as text (if completed)"
        )
        output_parsed: Optional[dict] = SchemaField(
            description="Structured JSON output matching outputSchema (if provided and completed)"
        )
        cost_total: Optional[float] = SchemaField(
            description="Total cost in USD (if completed)"
        )
        cost_searches: Optional[int] = SchemaField(
            description="Number of searches performed (if completed)"
        )
        cost_pages: Optional[int] = SchemaField(
            description="Number of pages crawled (if completed)"
        )
        cost_reasoning_tokens: Optional[int] = SchemaField(
            description="AI tokens used for reasoning (if completed)"
        )
        error_message: Optional[str] = SchemaField(
            description="Error message if research failed"
        )
        events: Optional[List[dict]] = SchemaField(
            description="Detailed event log (if include_events was True)"
        )

    def __init__(self):
        super().__init__(
            id="b2e3f4a5-6789-4bcd-9012-3456789abcde",
            description="Get status and results of a research task",
            categories={BlockCategory.SEARCH},
            input_schema=ExaGetResearchBlock.Input,
            output_schema=ExaGetResearchBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = f"https://api.exa.ai/research/v1/{input_data.research_id}"
        headers = {
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        params = {}
        if input_data.include_events:
            params["events"] = "true"

        response = await Requests().get(url, headers=headers, params=params)
        data = response.json()

        research = ResearchTaskModel.from_api(data)

        yield "research_id", research.research_id
        yield "status", research.status
        yield "instructions", research.instructions
        yield "model", research.model
        yield "created_at", research.created_at
        yield "finished_at", research.finished_at

        if research.output:
            yield "output_content", research.output.content
            yield "output_parsed", research.output.parsed

        if research.cost_dollars:
            yield "cost_total", research.cost_dollars.total
            yield "cost_searches", research.cost_dollars.num_searches
            yield "cost_pages", research.cost_dollars.num_pages
            yield "cost_reasoning_tokens", research.cost_dollars.reasoning_tokens

        yield "error_message", research.error

        if input_data.include_events:
            yield "events", data.get("events", [])


class ExaWaitForResearchBlock(Block):
    """Wait for a research task to complete with progress tracking."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        research_id: str = SchemaField(
            description="The ID of the research task to wait for",
            placeholder="01jszdfs0052sg4jc552sg4jc5",
        )
        timeout: int = SchemaField(
            default=600,
            description="Maximum time to wait in seconds",
            ge=1,
            le=3600,
        )
        check_interval: int = SchemaField(
            default=10,
            description="Seconds between status checks",
            advanced=True,
            ge=1,
            le=60,
        )

    class Output(BlockSchemaOutput):
        research_id: str = SchemaField(description="The research task identifier")
        final_status: str = SchemaField(description="Final status when polling stopped")
        output_content: Optional[str] = SchemaField(
            description="Research output as text (if completed)"
        )
        output_parsed: Optional[dict] = SchemaField(
            description="Structured JSON output (if outputSchema was provided and completed)"
        )
        cost_total: Optional[float] = SchemaField(description="Total cost in USD")
        elapsed_time: float = SchemaField(description="Total time waited in seconds")
        timed_out: bool = SchemaField(
            description="Whether polling timed out before completion"
        )

    def __init__(self):
        super().__init__(
            id="c3d4e5f6-7890-4abc-9012-3456789abcde",
            description="Wait for a research task to complete with configurable timeout",
            categories={BlockCategory.SEARCH},
            input_schema=ExaWaitForResearchBlock.Input,
            output_schema=ExaWaitForResearchBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        start_time = time.time()
        url = f"https://api.exa.ai/research/v1/{input_data.research_id}"
        headers = {
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        while time.time() - start_time < input_data.timeout:
            response = await Requests().get(url, headers=headers)
            data = response.json()

            status = data.get("status", "")

            if status in ["completed", "failed", "canceled"]:
                elapsed = time.time() - start_time
                research = ResearchTaskModel.from_api(data)

                yield "research_id", research.research_id
                yield "final_status", research.status
                yield "elapsed_time", elapsed
                yield "timed_out", False

                if research.output:
                    yield "output_content", research.output.content
                    yield "output_parsed", research.output.parsed

                if research.cost_dollars:
                    yield "cost_total", research.cost_dollars.total

                return

            await asyncio.sleep(input_data.check_interval)

        elapsed = time.time() - start_time
        response = await Requests().get(url, headers=headers)
        data = response.json()

        yield "research_id", input_data.research_id
        yield "final_status", data.get("status", "unknown")
        yield "elapsed_time", elapsed
        yield "timed_out", True


class ExaListResearchBlock(Block):
    """List all research tasks with pagination support."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        cursor: Optional[str] = SchemaField(
            default=None,
            description="Cursor for pagination through results",
            advanced=True,
        )
        limit: int = SchemaField(
            default=10,
            description="Number of research tasks to return (1-50)",
            ge=1,
            le=50,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        research_tasks: List[ResearchTaskModel] = SchemaField(
            description="List of research tasks ordered by creation time (newest first)"
        )
        research_task: ResearchTaskModel = SchemaField(
            description="Individual research task (yielded for each task)"
        )
        has_more: bool = SchemaField(
            description="Whether there are more tasks to paginate through"
        )
        next_cursor: Optional[str] = SchemaField(
            description="Cursor for the next page of results"
        )

    def __init__(self):
        super().__init__(
            id="d4e5f6a7-8901-4bcd-9012-3456789abcde",
            description="List all research tasks with pagination support",
            categories={BlockCategory.SEARCH},
            input_schema=ExaListResearchBlock.Input,
            output_schema=ExaListResearchBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = "https://api.exa.ai/research/v1"
        headers = {
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        params: Dict[str, Any] = {
            "limit": input_data.limit,
        }
        if input_data.cursor:
            params["cursor"] = input_data.cursor

        response = await Requests().get(url, headers=headers, params=params)
        data = response.json()

        tasks = [ResearchTaskModel.from_api(task) for task in data.get("data", [])]

        yield "research_tasks", tasks

        for task in tasks:
            yield "research_task", task

        yield "has_more", data.get("hasMore", False)
        yield "next_cursor", data.get("nextCursor")
