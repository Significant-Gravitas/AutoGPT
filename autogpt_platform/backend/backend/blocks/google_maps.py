from typing import List

import googlemaps

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import BlockSecret, SchemaField, SecretField


class GoogleMapsSearchBlock(Block):
    class Input(BlockSchema):
        api_key: BlockSecret = SecretField(
            key="google_maps_api_key",
            description="Google Maps API Key",
        )
        query: str = SchemaField(
            description="Search query",
            placeholder="e.g., 'New York'",
        )
        radius: int = SchemaField(
            description="Search radius in meters (max 50000)",
            default=5000,
            ge=1,
            le=50000,
        )
        max_results: int = SchemaField(
            description="Maximum number of places to return",
            default=5,
            ge=1,
            le=20,
        )

    class Output(BlockSchema):
        place: dict = SchemaField(description="A place found in the search")
        error: str = SchemaField(description="Error message if the search failed")

    def __init__(self):
        super().__init__(
            id="f47ac10b-58cc-4372-a567-0e02b2c3d479",
            description="This block searches for places using Google Maps API.",
            categories={BlockCategory.SEARCH},
            input_schema=GoogleMapsSearchBlock.Input,
            output_schema=GoogleMapsSearchBlock.Output,
            test_input={
                "api_key": "your_test_api_key",
                "query": "New York",
                "radius": 5000,
            },
            test_output=[
                (
                    "place",
                    {
                        "name": "New York",
                        "address": "New York, NY, USA",
                        "phone": "",
                        "rating": 0,
                        "reviews": 0,
                        "website": "http://www.nyc.gov/",
                    }
                ),
            ],
            test_mock={
                "search_places": lambda *args, **kwargs: [{
                    "name": "New York",
                    "address": "New York, NY, USA",
                    "phone": "",
                    "rating": 0,
                    "reviews": 0,
                    "website": "http://www.nyc.gov/",
                }]
            },
        )

    def run(self, input_data: Input) -> BlockOutput:
        try:
            places = self.search_places(
                input_data.api_key.get_secret_value(),
                input_data.query,
                input_data.radius,
                input_data.max_results,
            )
            for place in places:
                yield "place", place
        except Exception as e:
            yield "error", str(e)

    def search_places(self, api_key: str, query: str, radius: int, max_results: int) -> List[dict]:
        client = googlemaps.Client(key=api_key)
        result = client.places(query=query, radius=radius)
        
        if result['status'] == 'OK' and result['results']:
            places = []
            for place in result['results'][:max_results]:
                place_details = client.place(place['place_id'])['result']
                places.append({
                    "name": place_details.get('name', ''),
                    "address": place_details.get('formatted_address', ''),
                    "phone": place_details.get('formatted_phone_number', ''),
                    "rating": place_details.get('rating', 0),
                    "reviews": place_details.get('user_ratings_total', 0),
                    "website": place_details.get('website', '')
                })
            return places
        else:
            raise Exception("No results found or API error")
