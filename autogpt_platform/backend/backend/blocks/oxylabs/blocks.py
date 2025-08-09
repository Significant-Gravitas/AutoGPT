"""
Oxylabs Web Scraper API Blocks

This module implements blocks for interacting with the Oxylabs Web Scraper API.
Oxylabs provides powerful web scraping capabilities with anti-blocking measures,
JavaScript rendering, and built-in parsers for various sources.
"""

import base64
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from backend.sdk import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    Requests,
    SchemaField,
    UserPasswordCredentials,
)

from ._config import oxylabs


# Enums for Oxylabs API
class OxylabsSource(str, Enum):
    """Available scraping sources"""

    AMAZON_PRODUCT = "amazon_product"
    AMAZON_SEARCH = "amazon_search"
    GOOGLE_SEARCH = "google_search"
    GOOGLE_SHOPPING = "google_shopping"
    UNIVERSAL = "universal"
    # Add more sources as needed


class UserAgentType(str, Enum):
    """User agent types for scraping"""

    DESKTOP_CHROME = "desktop_chrome"
    DESKTOP_FIREFOX = "desktop_firefox"
    DESKTOP_SAFARI = "desktop_safari"
    DESKTOP_EDGE = "desktop_edge"
    MOBILE_ANDROID = "mobile_android"
    MOBILE_IOS = "mobile_ios"


class RenderType(str, Enum):
    """Rendering options"""

    NONE = "none"
    HTML = "html"
    PNG = "png"


class ResultType(str, Enum):
    """Result format types"""

    DEFAULT = "default"
    RAW = "raw"
    PARSED = "parsed"
    PNG = "png"


class JobStatus(str, Enum):
    """Job status values"""

    PENDING = "pending"
    DONE = "done"
    FAULTED = "faulted"


# Base class for Oxylabs blocks
class OxylabsBlockBase(Block):
    """Base class for all Oxylabs blocks with common functionality."""

    @staticmethod
    def get_auth_header(credentials: UserPasswordCredentials) -> str:
        """Create Basic Auth header from username and password."""
        username = credentials.username
        password = credentials.password.get_secret_value()
        auth_string = f"{username}:{password}"
        encoded = base64.b64encode(auth_string.encode()).decode()
        return f"Basic {encoded}"

    @staticmethod
    async def make_request(
        method: str,
        url: str,
        credentials: UserPasswordCredentials,
        json_data: Optional[dict] = None,
        params: Optional[dict] = None,
        timeout: int = 300,  # 5 minutes default for scraping
    ) -> dict:
        """Make an authenticated request to the Oxylabs API."""
        headers = {
            "Authorization": OxylabsBlockBase.get_auth_header(credentials),
            "Content-Type": "application/json",
        }

        response = await Requests().request(
            method=method,
            url=url,
            headers=headers,
            json=json_data,
            params=params,
            timeout=timeout,
        )

        if response.status < 200 or response.status >= 300:
            try:
                error_data = response.json()
            except Exception:
                error_data = {"message": response.text()}
            raise Exception(f"Oxylabs API error ({response.status}): {error_data}")

        # Handle empty responses (204 No Content)
        if response.status == 204:
            return {}

        return response.json()


# 1. Submit Job (Async)
class OxylabsSubmitJobAsyncBlock(OxylabsBlockBase):
    """
    Submit a scraping job asynchronously to Oxylabs.

    Returns a job ID for later polling or webhook delivery.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = oxylabs.credentials_field(
            description="Oxylabs username and password"
        )
        source: OxylabsSource = SchemaField(description="The source/site to scrape")
        url: Optional[str] = SchemaField(
            description="URL to scrape (for URL-based sources)", default=None
        )
        query: Optional[str] = SchemaField(
            description="Query/keyword/ID to search (for query-based sources)",
            default=None,
        )
        geo_location: Optional[str] = SchemaField(
            description="Geographical location (e.g., 'United States', '90210')",
            default=None,
        )
        parse: bool = SchemaField(
            description="Return structured JSON output", default=False
        )
        render: RenderType = SchemaField(
            description="Enable JS rendering or screenshots", default=RenderType.NONE
        )
        user_agent_type: Optional[UserAgentType] = SchemaField(
            description="User agent type for the request", default=None
        )
        callback_url: Optional[str] = SchemaField(
            description="Webhook URL for job completion notification", default=None
        )
        advanced_options: Optional[Dict[str, Any]] = SchemaField(
            description="Additional parameters (e.g., storage_type, context)",
            default=None,
        )

    class Output(BlockSchema):
        job_id: str = SchemaField(description="The Oxylabs job ID")
        status: str = SchemaField(description="Job status (usually 'pending')")
        self_url: str = SchemaField(description="URL to check job status")
        results_url: str = SchemaField(description="URL to get results (when done)")

    def __init__(self):
        super().__init__(
            id="a7c3b5d9-8e2f-4a1b-9c6d-3f7e8b9a0d5c",
            description="Submit an asynchronous scraping job to Oxylabs",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: UserPasswordCredentials, **kwargs
    ) -> BlockOutput:
        # Build request payload
        payload: Dict[str, Any] = {"source": input_data.source}

        # Add URL or query based on what's provided
        if input_data.url:
            payload["url"] = input_data.url
        elif input_data.query:
            payload["query"] = input_data.query
        else:
            raise ValueError("Either 'url' or 'query' must be provided")

        # Add optional parameters
        if input_data.geo_location:
            payload["geo_location"] = input_data.geo_location
        if input_data.parse:
            payload["parse"] = True
        if input_data.render != RenderType.NONE:
            payload["render"] = input_data.render
        if input_data.user_agent_type:
            payload["user_agent_type"] = input_data.user_agent_type
        if input_data.callback_url:
            payload["callback_url"] = input_data.callback_url

        # Merge advanced options
        if input_data.advanced_options:
            payload.update(input_data.advanced_options)

        # Submit job
        result = await self.make_request(
            method="POST",
            url="https://data.oxylabs.io/v1/queries",
            credentials=credentials,
            json_data=payload,
        )

        # Extract job info
        job_id = result.get("id", "")
        status = result.get("status", "pending")

        # Build URLs
        self_url = f"https://data.oxylabs.io/v1/queries/{job_id}"
        results_url = f"https://data.oxylabs.io/v1/queries/{job_id}/results"

        yield "job_id", job_id
        yield "status", status
        yield "self_url", self_url
        yield "results_url", results_url


# 2. Submit Job (Realtime)
class OxylabsSubmitJobRealtimeBlock(OxylabsBlockBase):
    """
    Submit a scraping job and wait for the result synchronously.

    The connection is held open until the scraping completes.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = oxylabs.credentials_field(
            description="Oxylabs username and password"
        )
        source: OxylabsSource = SchemaField(description="The source/site to scrape")
        url: Optional[str] = SchemaField(
            description="URL to scrape (for URL-based sources)", default=None
        )
        query: Optional[str] = SchemaField(
            description="Query/keyword/ID to search (for query-based sources)",
            default=None,
        )
        geo_location: Optional[str] = SchemaField(
            description="Geographical location (e.g., 'United States', '90210')",
            default=None,
        )
        parse: bool = SchemaField(
            description="Return structured JSON output", default=False
        )
        render: RenderType = SchemaField(
            description="Enable JS rendering or screenshots", default=RenderType.NONE
        )
        user_agent_type: Optional[UserAgentType] = SchemaField(
            description="User agent type for the request", default=None
        )
        advanced_options: Optional[Dict[str, Any]] = SchemaField(
            description="Additional parameters", default=None
        )

    class Output(BlockSchema):
        status: Literal["done", "faulted"] = SchemaField(
            description="Job completion status"
        )
        result: Union[str, dict, bytes] = SchemaField(
            description="Scraped content (HTML, JSON, or image)"
        )
        meta: Dict[str, Any] = SchemaField(description="Job metadata")

    def __init__(self):
        super().__init__(
            id="b8d4c6e0-9f3a-5b2c-0d7e-4a8f9c0b1e6d",
            description="Submit a synchronous scraping job to Oxylabs",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: UserPasswordCredentials, **kwargs
    ) -> BlockOutput:
        # Build request payload (similar to async, but no callback)
        payload: Dict[str, Any] = {"source": input_data.source}

        if input_data.url:
            payload["url"] = input_data.url
        elif input_data.query:
            payload["query"] = input_data.query
        else:
            raise ValueError("Either 'url' or 'query' must be provided")

        # Add optional parameters
        if input_data.geo_location:
            payload["geo_location"] = input_data.geo_location
        if input_data.parse:
            payload["parse"] = True
        if input_data.render != RenderType.NONE:
            payload["render"] = input_data.render
        if input_data.user_agent_type:
            payload["user_agent_type"] = input_data.user_agent_type

        # Merge advanced options
        if input_data.advanced_options:
            payload.update(input_data.advanced_options)

        # Submit job synchronously (using realtime endpoint)
        result = await self.make_request(
            method="POST",
            url="https://realtime.oxylabs.io/v1/queries",
            credentials=credentials,
            json_data=payload,
            timeout=600,  # 10 minutes for realtime
        )

        # Extract results
        status = "done" if result else "faulted"

        # Handle different result types
        content = result
        if input_data.parse and "results" in result:
            content = result["results"]
        elif "content" in result:
            content = result["content"]

        meta = {
            "source": input_data.source,
            "timestamp": datetime.utcnow().isoformat(),
        }

        yield "status", status
        yield "result", content
        yield "meta", meta


# 3. Submit Batch
class OxylabsSubmitBatchBlock(OxylabsBlockBase):
    """
    Submit multiple scraping jobs in one request (up to 5,000).

    Returns an array of job IDs for batch processing.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = oxylabs.credentials_field(
            description="Oxylabs username and password"
        )
        source: OxylabsSource = SchemaField(
            description="The source/site to scrape (applies to all)"
        )
        url_list: Optional[List[str]] = SchemaField(
            description="List of URLs to scrape", default=None
        )
        query_list: Optional[List[str]] = SchemaField(
            description="List of queries/keywords to search", default=None
        )
        geo_location: Optional[str] = SchemaField(
            description="Geographical location (applies to all)", default=None
        )
        parse: bool = SchemaField(
            description="Return structured JSON output", default=False
        )
        render: RenderType = SchemaField(
            description="Enable JS rendering or screenshots", default=RenderType.NONE
        )
        user_agent_type: Optional[UserAgentType] = SchemaField(
            description="User agent type for the requests", default=None
        )
        callback_url: Optional[str] = SchemaField(
            description="Webhook URL for job completion notifications", default=None
        )
        advanced_options: Optional[Dict[str, Any]] = SchemaField(
            description="Additional parameters", default=None
        )

    class Output(BlockSchema):
        job_ids: List[str] = SchemaField(description="List of job IDs")
        count: int = SchemaField(description="Number of jobs created")

    def __init__(self):
        super().__init__(
            id="c9e5d7f1-0a4b-6c3d-1e8f-5b9a0c2d3f7e",
            description="Submit batch scraping jobs to Oxylabs",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: UserPasswordCredentials, **kwargs
    ) -> BlockOutput:
        # Build batch request payload
        payload: Dict[str, Any] = {"source": input_data.source}

        # Add URL list or query list
        if input_data.url_list:
            if len(input_data.url_list) > 5000:
                raise ValueError("Batch size cannot exceed 5,000 URLs")
            payload["url"] = input_data.url_list
        elif input_data.query_list:
            if len(input_data.query_list) > 5000:
                raise ValueError("Batch size cannot exceed 5,000 queries")
            payload["query"] = input_data.query_list
        else:
            raise ValueError("Either 'url_list' or 'query_list' must be provided")

        # Add optional parameters (apply to all items)
        if input_data.geo_location:
            payload["geo_location"] = input_data.geo_location
        if input_data.parse:
            payload["parse"] = True
        if input_data.render != RenderType.NONE:
            payload["render"] = input_data.render
        if input_data.user_agent_type:
            payload["user_agent_type"] = input_data.user_agent_type
        if input_data.callback_url:
            payload["callback_url"] = input_data.callback_url

        # Merge advanced options
        if input_data.advanced_options:
            payload.update(input_data.advanced_options)

        # Submit batch
        result = await self.make_request(
            method="POST",
            url="https://data.oxylabs.io/v1/queries/batch",
            credentials=credentials,
            json_data=payload,
        )

        # Extract job IDs
        queries = result.get("queries", [])
        job_ids = [q.get("id", "") for q in queries if q.get("id")]

        yield "job_ids", job_ids
        yield "count", len(job_ids)


# 4. Check Job Status
class OxylabsCheckJobStatusBlock(OxylabsBlockBase):
    """
    Check the status of a scraping job.

    Can optionally wait for completion by polling.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = oxylabs.credentials_field(
            description="Oxylabs username and password"
        )
        job_id: str = SchemaField(description="Job ID to check")
        wait_for_completion: bool = SchemaField(
            description="Poll until job leaves 'pending' status", default=False
        )

    class Output(BlockSchema):
        status: JobStatus = SchemaField(description="Current job status")
        updated_at: Optional[str] = SchemaField(
            description="Last update timestamp", default=None
        )
        results_url: Optional[str] = SchemaField(
            description="URL to get results (when done)", default=None
        )
        raw_status: Dict[str, Any] = SchemaField(description="Full status response")

    def __init__(self):
        super().__init__(
            id="d0f6e8a2-1b5c-7d4e-2f9a-6c0b1d3e4a8f",
            description="Check the status of an Oxylabs scraping job",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: UserPasswordCredentials, **kwargs
    ) -> BlockOutput:
        import asyncio

        url = f"https://data.oxylabs.io/v1/queries/{input_data.job_id}"

        # Check status (with optional polling)
        max_attempts = 60 if input_data.wait_for_completion else 1
        delay = 5  # seconds between polls

        # Initialize variables that will be used outside the loop
        result = {}
        status = "pending"

        for attempt in range(max_attempts):
            result = await self.make_request(
                method="GET",
                url=url,
                credentials=credentials,
            )

            status = result.get("status", "pending")

            # If not waiting or job is complete, return
            if not input_data.wait_for_completion or status != "pending":
                break

            # Wait before next poll
            if attempt < max_attempts - 1:
                await asyncio.sleep(delay)

        # Extract results URL if job is done
        results_url = None
        if status == "done":
            links = result.get("_links", [])
            for link in links:
                if link.get("rel") == "results":
                    results_url = link.get("href")
                    break

        yield "status", JobStatus(status)
        yield "updated_at", result.get("updated_at")
        yield "results_url", results_url
        yield "raw_status", result


# 5. Get Job Results
class OxylabsGetJobResultsBlock(OxylabsBlockBase):
    """
    Download the scraped data for a completed job.

    Supports different result formats (raw, parsed, screenshot).
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = oxylabs.credentials_field(
            description="Oxylabs username and password"
        )
        job_id: str = SchemaField(description="Job ID to get results for")
        result_type: ResultType = SchemaField(
            description="Type of result to retrieve", default=ResultType.DEFAULT
        )

    class Output(BlockSchema):
        content: Union[str, dict, bytes] = SchemaField(description="The scraped data")
        content_type: str = SchemaField(description="MIME type of the content")
        meta: Dict[str, Any] = SchemaField(description="Result metadata")

    def __init__(self):
        super().__init__(
            id="e1a7f9b3-2c6d-8e5f-3a0b-7d1c2e4f5b9a",
            description="Get results from a completed Oxylabs job",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: UserPasswordCredentials, **kwargs
    ) -> BlockOutput:
        url = f"https://data.oxylabs.io/v1/queries/{input_data.job_id}/results"

        # Add result type parameter if not default
        params = {}
        if input_data.result_type != ResultType.DEFAULT:
            params["type"] = input_data.result_type

        # Get results
        headers = {
            "Authorization": self.get_auth_header(credentials),
        }

        # For PNG results, we need to handle binary data
        if input_data.result_type == ResultType.PNG:
            response = await Requests().request(
                method="GET",
                url=url,
                headers=headers,
                params=params,
            )

            if response.status < 200 or response.status >= 300:
                raise Exception(f"Failed to get results: {response.status}")

            content = response.content  # Binary content
            content_type = response.headers.get("Content-Type", "image/png")
        else:
            # JSON or text results
            result = await self.make_request(
                method="GET",
                url=url,
                credentials=credentials,
                params=params,
            )

            content = result
            content_type = "application/json"

        meta = {
            "job_id": input_data.job_id,
            "result_type": input_data.result_type,
            "retrieved_at": datetime.utcnow().isoformat(),
        }

        yield "content", content
        yield "content_type", content_type
        yield "meta", meta


# 6. Proxy Fetch URL
class OxylabsProxyFetchBlock(OxylabsBlockBase):
    """
    Fetch a URL through Oxylabs' HTTPS proxy endpoint.

    Ideal for one-off page downloads without job management.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = oxylabs.credentials_field(
            description="Oxylabs username and password"
        )
        target_url: str = SchemaField(
            description="URL to fetch (must include https://)"
        )
        geo_location: Optional[str] = SchemaField(
            description="Geographical location", default=None
        )
        user_agent_type: Optional[UserAgentType] = SchemaField(
            description="User agent type", default=None
        )
        render: Literal["none", "html"] = SchemaField(
            description="Enable JavaScript rendering", default="none"
        )

    class Output(BlockSchema):
        html: str = SchemaField(description="Page HTML content")
        status_code: int = SchemaField(description="HTTP status code")
        headers: Dict[str, str] = SchemaField(description="Response headers")

    def __init__(self):
        super().__init__(
            id="f2b8a0c4-3d7e-9f6a-4b1c-8e2d3f5a6c0b",
            description="Fetch a URL through Oxylabs proxy",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: UserPasswordCredentials, **kwargs
    ) -> BlockOutput:
        # Prepare proxy headers
        headers = {
            "Authorization": self.get_auth_header(credentials),
        }

        if input_data.geo_location:
            headers["x-oxylabs-geo-location"] = input_data.geo_location
        if input_data.user_agent_type:
            headers["x-oxylabs-user-agent-type"] = input_data.user_agent_type
        if input_data.render != "none":
            headers["x-oxylabs-render"] = input_data.render

        # Use the proxy endpoint
        # Note: In a real implementation, you'd configure the HTTP client
        # to use realtime.oxylabs.io:60000 as an HTTPS proxy
        # For this example, we'll use the regular API endpoint

        payload = {
            "source": "universal",
            "url": input_data.target_url,
        }

        if input_data.geo_location:
            payload["geo_location"] = input_data.geo_location
        if input_data.user_agent_type:
            payload["user_agent_type"] = input_data.user_agent_type
        if input_data.render != "none":
            payload["render"] = input_data.render

        result = await self.make_request(
            method="POST",
            url="https://realtime.oxylabs.io/v1/queries",
            credentials=credentials,
            json_data=payload,
            timeout=300,
        )

        # Extract content
        html = result.get("content", "")
        status_code = result.get("status_code", 200)
        headers = result.get("headers", {})

        yield "html", html
        yield "status_code", status_code
        yield "headers", headers


# 7. Callback Trigger (Webhook) - This would be handled by the platform's webhook system
# We'll create a block to process webhook data instead
class OxylabsProcessWebhookBlock(OxylabsBlockBase):
    """
    Process incoming Oxylabs webhook callback data.

    Extracts job information from the webhook payload.
    """

    class Input(BlockSchema):
        webhook_payload: Dict[str, Any] = SchemaField(
            description="Raw webhook payload from Oxylabs"
        )
        verify_ip: bool = SchemaField(
            description="Verify the request came from Oxylabs IPs", default=True
        )
        source_ip: Optional[str] = SchemaField(
            description="IP address of the webhook sender", default=None
        )

    class Output(BlockSchema):
        job_id: str = SchemaField(description="Job ID from callback")
        status: JobStatus = SchemaField(description="Job completion status")
        results_url: Optional[str] = SchemaField(
            description="URL to fetch the results", default=None
        )
        raw_callback: Dict[str, Any] = SchemaField(description="Full callback payload")

    def __init__(self):
        super().__init__(
            id="a3c9b1d5-4e8f-0b2d-5c6e-9f0a1d3f7b8c",
            description="Process Oxylabs webhook callback data",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: UserPasswordCredentials, **kwargs
    ) -> BlockOutput:
        payload = input_data.webhook_payload

        # Extract job information
        job_id = payload.get("id", "")
        status = JobStatus(payload.get("status", "pending"))

        # Find results URL
        results_url = None
        links = payload.get("_links", [])
        for link in links:
            if link.get("rel") == "results":
                results_url = link.get("href")
                break

        # If IP verification is requested, we'd check against the callbacker IPs
        # This is simplified for the example
        if input_data.verify_ip and input_data.source_ip:
            # In a real implementation, we'd fetch and cache the IP list
            # and verify the source_ip is in that list
            pass

        yield "job_id", job_id
        yield "status", status
        yield "results_url", results_url
        yield "raw_callback", payload


# 8. Callbacker IP List
class OxylabsCallbackerIPListBlock(OxylabsBlockBase):
    """
    Get the list of IP addresses used by Oxylabs for callbacks.

    Use this for firewall whitelisting.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = oxylabs.credentials_field(
            description="Oxylabs username and password"
        )

    class Output(BlockSchema):
        ip_list: List[str] = SchemaField(description="List of Oxylabs callback IPs")
        updated_at: str = SchemaField(description="Timestamp of retrieval")

    def __init__(self):
        super().__init__(
            id="b4d0c2e6-5f9a-1c3e-6d7f-0a1b2d4e8c9d",
            description="Get Oxylabs callback IP addresses",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: UserPasswordCredentials, **kwargs
    ) -> BlockOutput:
        result = await self.make_request(
            method="GET",
            url="https://data.oxylabs.io/v1/info/callbacker_ips",
            credentials=credentials,
        )

        # Extract IP list
        ip_list = result.get("callbacker_ips", [])
        updated_at = datetime.utcnow().isoformat()

        yield "ip_list", ip_list
        yield "updated_at", updated_at
