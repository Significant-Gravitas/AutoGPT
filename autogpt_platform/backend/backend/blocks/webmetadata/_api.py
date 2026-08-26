"""
Thin client for the Web Metadata Extractor API's RapidAPI gateway, shared by
all blocks in this provider.
"""

from backend.sdk import APIKeyCredentials, Requests

RAPIDAPI_HOST = "web-metadata-and-contact-extractor.p.rapidapi.com"
BASE_URL = f"https://{RAPIDAPI_HOST}"


async def call_endpoint(credentials: APIKeyCredentials, path: str, url: str) -> dict:
    """GET one of the API's /api/v1/* endpoints for the given target URL."""
    response = await Requests().get(
        f"{BASE_URL}{path}",
        headers={
            "X-RapidAPI-Key": credentials.api_key.get_secret_value(),
            "X-RapidAPI-Host": RAPIDAPI_HOST,
        },
        params={"url": url},
    )
    return response.json()
