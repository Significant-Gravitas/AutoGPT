from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    NodeExecutionStats,
    SchemaField,
)
from backend.integrations.providers import ProviderName
from backend.util.request import Requests, Response

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="google_maps",
    api_key=SecretStr("mock-google-maps-api-key"),
    title="Mock Google Maps API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.type,
}

SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
PAGE_SIZE = 20
# Text Search bills each page by the most expensive field requested: the
# phone, website, rating and review count make it the Enterprise rate.
SEARCH_FIELDS = ",".join(
    [
        "places.id",
        "places.displayName",
        "places.formattedAddress",
        "places.location",
        "places.googleMapsUri",
        "places.nationalPhoneNumber",
        "places.websiteUri",
        "places.rating",
        "places.userRatingCount",
        "nextPageToken",
    ]
)
_ERROR_HINTS = {
    "API_KEY_INVALID": (
        "Google rejected the Maps API key. Check the key saved in your Google "
        "Maps credentials."
    ),
    "SERVICE_DISABLED": (
        "Places API (New) isn't enabled on the Google Cloud project that owns "
        "this Maps API key. Enable it in the Google Cloud console under APIs & "
        "Services, then try again."
    ),
    "API_KEY_SERVICE_BLOCKED": (
        "This Maps API key isn't allowed to call Places API (New). Add it to "
        "the key's API restrictions in the Google Cloud console."
    ),
}


class Place(BaseModel):
    name: str
    address: str
    phone: str
    rating: float
    reviews: int
    website: str
    place_id: str = Field(default="", description="Google Maps place ID")
    latitude: float | None = Field(default=None, description="Latitude in degrees")
    longitude: float | None = Field(default=None, description="Longitude in degrees")
    google_maps_url: str = Field(
        default="", description="Link to the place on Google Maps"
    )


class GoogleMapsSearchBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.GOOGLE_MAPS], Literal["api_key"]
        ] = CredentialsField(description="Google Maps API Key")
        query: str = SchemaField(
            description="Search query for local businesses",
            placeholder="e.g., 'restaurants in New York'",
        )
        radius: int = SchemaField(
            description=(
                "Not used: Google needs a centre point to apply a radius, so "
                "name the area in the query instead. Kept so existing agents "
                "still load."
            ),
            default=5000,
            ge=1,
            le=50000,
            advanced=True,
        )
        max_results: int = SchemaField(
            description="Maximum number of results to return (max 60)",
            default=20,
            ge=1,
            le=60,
        )

    class Output(BlockSchemaOutput):
        place: Place = SchemaField(description="Place found")

    def __init__(self):
        super().__init__(
            id="f47ac10b-58cc-4372-a567-0e02b2c3d479",
            description=(
                "Search Google Maps for businesses and other places that match a "
                "text query. Returns each place's name, address, phone, rating, "
                "review count, website, place ID, coordinates and Google Maps link."
            ),
            categories={BlockCategory.SEARCH},
            input_schema=GoogleMapsSearchBlock.Input,
            output_schema=GoogleMapsSearchBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "query": "restaurants in new york",
                "radius": 5000,
                "max_results": 5,
            },
            test_output=[
                (
                    "place",
                    {
                        "name": "Test Restaurant",
                        "address": "123 Test St, New York, NY 10001",
                        "phone": "(555) 123-4567",
                        "rating": 4.5,
                        "reviews": 100,
                        "website": "https://testrestaurant.com",
                        "place_id": "ChIJTestRestaurant0123456789",
                        "latitude": 40.7505,
                        "longitude": -73.9934,
                        "google_maps_url": "https://maps.google.com/?cid=1234567890",
                    },
                ),
            ],
            test_mock={
                "search_places": lambda *args, **kwargs: [
                    {
                        "name": "Test Restaurant",
                        "address": "123 Test St, New York, NY 10001",
                        "phone": "(555) 123-4567",
                        "rating": 4.5,
                        "reviews": 100,
                        "website": "https://testrestaurant.com",
                        "place_id": "ChIJTestRestaurant0123456789",
                        "latitude": 40.7505,
                        "longitude": -73.9934,
                        "google_maps_url": "https://maps.google.com/?cid=1234567890",
                    }
                ]
            },
            test_credentials=TEST_CREDENTIALS,
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        places = await self.search_places(
            credentials.api_key,
            input_data.query,
            input_data.radius,
            input_data.max_results,
        )
        self.merge_stats(
            NodeExecutionStats(
                provider_cost=float(len(places)), provider_cost_type="items"
            )
        )
        for place in places:
            yield "place", place

    async def search_places(
        self, api_key: SecretStr, query: str, radius: int, max_results: int
    ) -> list[Place]:
        """Search Places API (New), a page of up to 20 places at a time.

        `radius` is accepted for existing agents, but a radius needs a centre
        point, which the block doesn't have, so it isn't sent.
        """
        places: list[Place] = []
        page_token = ""
        while len(places) < max_results:
            response = await self._search_page(
                api_key, query, min(PAGE_SIZE, max_results - len(places)), page_token
            )
            places.extend(to_place(place) for place in response.get("places") or [])
            page_token = response.get("nextPageToken", "")
            if not page_token:
                break
        return places[:max_results]

    @staticmethod
    async def _search_page(
        api_key: SecretStr, query: str, page_size: int, page_token: str
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"textQuery": query, "pageSize": page_size}
        if page_token:
            body["pageToken"] = page_token
        response = await Requests(raise_for_status=False, retry_max_attempts=3).post(
            SEARCH_URL,
            headers={
                "X-Goog-Api-Key": api_key.get_secret_value(),
                "X-Goog-FieldMask": SEARCH_FIELDS,
            },
            json=body,
            allow_redirects=False,
        )
        if not response.ok:
            raise ValueError(search_error(response))
        return response.json()


def to_place(place: dict[str, Any]) -> Place:
    """Map a Places API (New) place to the block's Place output."""
    location = place.get("location") or {}
    return Place(
        name=(place.get("displayName") or {}).get("text", ""),
        address=place.get("formattedAddress", ""),
        phone=place.get("nationalPhoneNumber", ""),
        rating=place.get("rating", 0),
        reviews=place.get("userRatingCount", 0),
        website=place.get("websiteUri", ""),
        place_id=place.get("id", ""),
        latitude=location.get("latitude", 0.0) if location else None,
        longitude=location.get("longitude", 0.0) if location else None,
        google_maps_url=place.get("googleMapsUri", ""),
    )


def search_error(response: Response) -> str:
    """A message the user can act on, for a failed Text Search request."""
    body = response.json(fallback={}) if response.content else {}
    error = (body.get("error") if isinstance(body, dict) else None) or {}
    for detail in error.get("details") or []:
        if hint := _ERROR_HINTS.get(detail.get("reason", "")):
            return hint
    if response.status == 429:
        return (
            "This Maps API key has run out of Places API (New) quota. Try again "
            "later, or raise the quota in the Google Cloud console."
        )
    if response.status >= 500:
        return (
            f"Places API (New) had a temporary problem (HTTP {response.status}). "
            "Try again."
        )
    detail = error.get("message") or response.reason or "no details given"
    return f"Places API (New) rejected the search (HTTP {response.status}): {detail}"
