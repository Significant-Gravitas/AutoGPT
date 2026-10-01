from datetime import datetime
from typing import Any

from pydantic import SecretStr

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks._google_maps_api import (
    GoogleMapsCredentialsField,
    GoogleMapsCredentialsInput,
    GoogleMapsError,
    UnitsSystem,
)
from backend.blocks._google_maps_routes_api import (
    MODE_WORDS,
    TRAFFIC_MODES,
    RouteStep,
    TravelMode,
    build_route_request,
    compute_route,
    route_outputs,
    to_route_step,
)
from backend.blocks.google_maps import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.data.model import APIKeyCredentials, NodeExecutionStats, SchemaField
from backend.util.exceptions import BlockExecutionError, BlockInputError


def _test_step(meters: int, seconds: int, maneuver: str, text: str) -> dict:
    return {
        "distanceMeters": meters,
        "staticDuration": f"{seconds}s",
        "navigationInstruction": {"maneuver": maneuver, "instructions": text},
        "localizedValues": {
            "distance": {"text": f"{meters / 1000:.1f} km"},
            "staticDuration": {"text": f"{max(1, round(seconds / 60))} min"},
        },
        "travelMode": "DRIVE",
    }


_TEST_ROUTE = {
    "distanceMeters": 4627,
    "duration": "868s",
    "description": "Voie Georges Pompidou",
    "localizedValues": {
        "distance": {"text": "4.6 km"},
        "duration": {"text": "14 mins"},
    },
    "legs": [
        {
            "steps": [
                _test_step(250, 52, "DEPART", "Head northeast on Quai Branly"),
                _test_step(
                    4377, 816, "STRAIGHT", "Continue onto Voie Georges Pompidou"
                ),
            ]
        }
    ],
}


class GoogleMapsGetDirectionsBlock(Block):
    """Distance, travel time and directions between two places."""

    class Input(BlockSchemaInput):
        credentials: GoogleMapsCredentialsInput = GoogleMapsCredentialsField()
        origin: str = SchemaField(
            description=(
                "Where the route starts: an address, a place name, "
                "'latitude,longitude' or a Google Maps place ID"
            ),
            placeholder="e.g. 'Eiffel Tower, Paris'",
        )
        destination: str = SchemaField(
            description="Where the route ends, in the same forms as the origin",
            placeholder="e.g. 'Louvre Museum, Paris'",
        )
        travel_mode: TravelMode = SchemaField(
            description=(
                "drive, walk, bicycle, transit (public transport) or two_wheeler "
                "(motorbikes and scooters, only in some countries). Google bills "
                "two-wheeler routes at a higher rate."
            ),
            default=TravelMode.DRIVE,
        )
        include_steps: bool = SchemaField(
            description="Also return turn-by-turn steps", default=False
        )
        use_live_traffic: bool = SchemaField(
            description=(
                "Use live traffic for drive and two-wheeler routes, for more "
                "accurate travel times. Google bills these at a higher rate."
            ),
            default=False,
        )
        units: UnitsSystem = SchemaField(
            description="metric (km) or imperial (miles), for the distance and time text",
            default=UnitsSystem.METRIC,
        )
        departure_time: datetime | None = SchemaField(
            description=(
                "When to leave, for transit timetables or live-traffic "
                "predictions. Defaults to now. A time without a time zone is "
                "read as UTC."
            ),
            default=None,
            advanced=True,
        )
        language_code: str = SchemaField(
            description="Language for the directions, e.g. 'en' or 'fr'",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        summary: str = SchemaField(
            description="Travel time, distance and main roads, e.g. '14 mins (4.6 km) via Voie Georges Pompidou'"
        )
        distance_meters: int = SchemaField(description="Length of the route in meters")
        distance_text: str = SchemaField(
            description="Length of the route as text, e.g. '4.6 km' or '2.9 mi'"
        )
        duration_seconds: int = SchemaField(
            description="Travel time in seconds, including traffic when live traffic is on"
        )
        duration_text: str = SchemaField(
            description="Travel time as text, e.g. '14 mins'"
        )
        steps: list[RouteStep] = SchemaField(
            description="Turn-by-turn steps, when include_steps is on"
        )
        step: RouteStep = SchemaField(description="Each step, when include_steps is on")
        warnings: list[str] = SchemaField(
            description="Warnings to show with the route, e.g. that walking directions are in beta"
        )

    def __init__(self):
        route_facts = [
            ("summary", "14 mins (4.6 km) via Voie Georges Pompidou"),
            ("distance_meters", 4627),
            ("distance_text", "4.6 km"),
            ("duration_seconds", 868),
            ("duration_text", "14 mins"),
        ]
        test_steps = [to_route_step(step) for step in _TEST_ROUTE["legs"][0]["steps"]]
        super().__init__(
            id="61cbd1a4-f844-408f-9646-db12662209aa",
            description=(
                "Get directions between two places with Google Maps: distance, "
                "travel time and a route summary, plus turn-by-turn steps if you "
                "ask for them. Works for driving, walking, cycling, public "
                "transport and two-wheelers, with optional live traffic."
            ),
            categories={BlockCategory.SEARCH},
            input_schema=GoogleMapsGetDirectionsBlock.Input,
            output_schema=GoogleMapsGetDirectionsBlock.Output,
            test_input=[
                {
                    "credentials": TEST_CREDENTIALS_INPUT,
                    "origin": "Eiffel Tower, Paris",
                    "destination": "Louvre Museum, Paris",
                },
                {
                    "credentials": TEST_CREDENTIALS_INPUT,
                    "origin": "48.8583701,2.2944813",
                    "destination": "ChIJD3uTd9hx5kcR1IQvGfr8dbk",
                    "include_steps": True,
                },
            ],
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                *route_facts,
                ("warnings", []),
                *route_facts,
                ("steps", test_steps),
                *[("step", step) for step in test_steps],
                ("warnings", []),
            ],
            test_mock={
                "_compute_route": lambda *args, **kwargs: {"routes": [_TEST_ROUTE]}
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        self._check_input(input_data)
        body = build_route_request(
            origin=input_data.origin,
            destination=input_data.destination,
            travel_mode=input_data.travel_mode,
            use_live_traffic=input_data.use_live_traffic,
            units=input_data.units,
            language_code=input_data.language_code,
            departure_time=input_data.departure_time,
        )
        try:
            response = await self._compute_route(
                credentials.api_key, body, input_data.include_steps
            )
        except GoogleMapsError as e:
            raise BlockExecutionError(
                message=str(e), block_name=self.name, block_id=self.id
            ) from e
        self.merge_stats(
            NodeExecutionStats(provider_cost=1.0, provider_cost_type="items")
        )
        routes = response.get("routes") or []
        if not routes:
            raise BlockExecutionError(
                message=(
                    f"Google Maps found no {MODE_WORDS[input_data.travel_mode]} "
                    f"route from '{input_data.origin.strip()}' to "
                    f"'{input_data.destination.strip()}'."
                ),
                block_name=self.name,
                block_id=self.id,
            )
        for name, value in route_outputs(
            routes[0], include_steps=input_data.include_steps
        ):
            yield name, value

    def _check_input(self, input_data: Input) -> None:
        if not input_data.origin.strip() or not input_data.destination.strip():
            raise BlockInputError(
                message="Give both an origin and a destination.",
                block_name=self.name,
                block_id=self.id,
            )
        if input_data.use_live_traffic and input_data.travel_mode not in TRAFFIC_MODES:
            raise BlockInputError(
                message=(
                    "Live traffic only applies to drive and two-wheeler routes. "
                    "Turn it off for walking, cycling or transit."
                ),
                block_name=self.name,
                block_id=self.id,
            )

    @staticmethod
    async def _compute_route(
        api_key: SecretStr, body: dict[str, Any], include_steps: bool
    ) -> dict[str, Any]:
        return await compute_route(api_key, body, include_steps=include_steps)
