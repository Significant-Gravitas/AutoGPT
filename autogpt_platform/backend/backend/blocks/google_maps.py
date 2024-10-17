import googlemaps
from pydantic import BaseModel

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import BlockSecret, SchemaField, SecretField


class Place(BaseModel):
    name: str
    address: str
    phone: str
    rating: float
    reviews: int
    website: str


class GoogleMapsSearchBlock(Block):
    class Input(BlockSchema):
        api_key: BlockSecret = SecretField(
            key="google_maps_api_key",
            description="Google Maps API Key",
        )
        query: str = SchemaField(
            description="Search query for local businesses",
            placeholder="e.g., 'restaurants in New York'",
        )
        radius: int = SchemaField(
            description="Search radius in meters (max 50000)",
            default=5000,
            ge=1,
            le=50000,
        )
        max_results: int = SchemaField(
            description="Maximum number of results to return (max 60)",
            default=20,
            ge=1,
            le=60,
        )

    class Output(BlockSchema):
        place: Place = SchemaField(description="Place found")
        error: str = SchemaField(description="Error message if the search failed")

    def __init__(self):
        super().__init__(
            id="f47ac10b-58cc-4372-a567-0e02b2c3d479",
            description="This block searches for local businesses using Google Maps API.",
            categories={BlockCategory.SEARCH},
            input_schema=GoogleMapsSearchBlock.Input,
            output_schema=GoogleMapsSearchBlock.Output,
            test_input={
                "api_key": "your_test_api_key",
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
                        "phone": "+1 (555) 123-4567",
                        "rating": 4.5,
                        "reviews": 100,
                        "website": "https://testrestaurant.com",
                    },
                ),
            ],
            test_mock={
                "search_places": lambda *args, **kwargs: [
                    {
                        "name": "Test Restaurant",
                        "address": "123 Test St, New York, NY 10001",
                        "phone": "+1 (555) 123-4567",
                        "rating": 4.5,
                        "reviews": 100,
                        "website": "https://testrestaurant.com",
                    }
                ]
            },
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        places = self.search_places(
            input_data.api_key.get_secret_value(),
            input_data.query,
            input_data.radius,
            input_data.max_results,
        )
        for place in places:
            yield "place", place

    def search_places(self, api_key, query, radius, max_results):
        client = googlemaps.Client(key=api_key)
        return self._search_places(client, query, radius, max_results)

    def _search_places(self, client, query, radius, max_results):
        results = []
        next_page_token = None
        while len(results) < max_results:
            response = client.places(
                query=query,
                radius=radius,
                page_token=next_page_token,
            )
            for place in response["results"]:
                if len(results) >= max_results:
                    break
                place_details = client.place(place["place_id"])["result"]
                results.append(
                    Place(
                        name=place_details.get("name", ""),
                        address=place_details.get("formatted_address", ""),
                        phone=place_details.get("formatted_phone_number", ""),
                        rating=place_details.get("rating", 0),
                        reviews=place_details.get("user_ratings_total", 0),
                        website=place_details.get("website", ""),
                    )
                )
            next_page_token = response.get("next_page_token")
            if not next_page_token:
                break
        return results
