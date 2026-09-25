import asyncio
from typing import Any

from pydantic import BaseModel, Field, SecretStr

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks._google_maps_api import (
    PLACE_PRO_FIELDS,
    GoogleMapsCredentialsField,
    GoogleMapsCredentialsInput,
    GoogleMapsError,
    MapsPlace,
    get_place,
    parse_place_id,
    place_fields,
    search_place_id,
)
from backend.blocks._google_maps_links import (
    LinkError,
    LinkTarget,
    expand_link,
    find_link_place,
    parse_link,
)
from backend.blocks.google_maps import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.data.model import APIKeyCredentials, NodeExecutionStats, SchemaField
from backend.util.exceptions import BlockExecutionError


class ResolvedPlace(MapsPlace):
    """A place matched to one of the queries."""

    query: str = Field(description="The query this place was found for")


class ResolvedMapsLink(MapsPlace):
    """The place, or the point on the map, that a link points at."""

    url: str = Field(description="The link as given")
    expanded_url: str = Field(
        default="", description="The full Google Maps URL, after any short link"
    )


class FailedMapsLink(BaseModel):
    """A link that couldn't be resolved."""

    url: str = Field(description="The link as given")
    reason: str = Field(description="Why it couldn't be resolved")


class _LinkOutcome(BaseModel):
    result: ResolvedMapsLink | None = None
    failure: FailedMapsLink | None = None
    looked_up: bool = False


_TEST_PLACE = {
    "id": "ChIJLU7jZClu5kcR4PcOOO6p3I0",
    "displayName": {"text": "Eiffel Tower", "languageCode": "en"},
    "formattedAddress": "Av. Gustave Eiffel, 75007 Paris, France",
    "location": {"latitude": 48.8583701, "longitude": 2.2944813},
    "types": ["tourist_attraction", "point_of_interest", "establishment"],
    "googleMapsUri": "https://maps.google.com/?cid=10222232094831998944",
}
_TEST_SHORT_LINK = "https://maps.app.goo.gl/Xj9kLmNoPqRsTuVw8"
_TEST_DIRECTIONS_LINK = "https://www.google.com/maps/dir/Paris/Lyon"
_TEST_EXPANDED_LINK = (
    "https://www.google.com/maps/place/Eiffel+Tower/@48.8583701,2.2944813,17z/"
    "data=!3m1!4b1!4m6!3m5!1s0x47e66e2964e34e2d:0x8ddca9ee380ef7e0!8m2"
    "!3d48.8583701!4d2.2944813!16zL20vMDJqODE?entry=ttu"
)


class GoogleMapsResolvePlacesBlock(Block):
    """Match place names or addresses to Google Maps places."""

    class Input(BlockSchemaInput):
        credentials: GoogleMapsCredentialsInput = GoogleMapsCredentialsField()
        queries: list[str] = SchemaField(
            description=(
                "Place names or addresses to look up, up to 20. Be specific, e.g. "
                "'Eiffel Tower, Paris' or '1600 Amphitheatre Pkwy, Mountain View, CA'. "
                "Searches like 'coffee near me' or chain names like 'Starbucks' "
                "don't name one place."
            ),
            min_length=1,
            max_length=20,
        )
        region_code: str = SchemaField(
            description="Two-letter country code to prefer matches in, e.g. 'US' or 'GB'",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        places: list[ResolvedPlace] = SchemaField(
            description="The places found, in the order of the queries"
        )
        place: ResolvedPlace = SchemaField(description="Each place found")
        unresolved: list[str] = SchemaField(description="Queries that matched no place")

    def __init__(self):
        test_place = ResolvedPlace(
            query="Eiffel Tower, Paris", **place_fields(_TEST_PLACE)
        )
        super().__init__(
            id="2777bfe3-da96-465f-88d9-d3cb4621f15d",
            description=(
                "Look up place names or addresses on Google Maps and get each "
                "one's place ID, name, full address, coordinates, types and "
                "Google Maps link. Takes up to 20 at once."
            ),
            categories={BlockCategory.SEARCH},
            input_schema=GoogleMapsResolvePlacesBlock.Input,
            output_schema=GoogleMapsResolvePlacesBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "queries": ["Eiffel Tower, Paris", "Nowhere Special 12345"],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("places", [test_place]),
                ("place", test_place),
                ("unresolved", ["Nowhere Special 12345"]),
            ],
            test_mock={
                "_resolve_query": lambda api_key, query, region_code: (
                    _TEST_PLACE if query.startswith("Eiffel") else None
                )
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        queries = [query.strip() for query in input_data.queries]
        try:
            found = await asyncio.gather(
                *(
                    self._resolve_query(
                        credentials.api_key, query, input_data.region_code
                    )
                    for query in queries
                )
            )
        except GoogleMapsError as e:
            raise BlockExecutionError(
                message=str(e), block_name=self.name, block_id=self.id
            ) from e
        places = [
            ResolvedPlace(query=query, **place_fields(place))
            for query, place in zip(queries, found)
            if place
        ]
        self.merge_stats(
            NodeExecutionStats(
                provider_cost=float(len(places)), provider_cost_type="items"
            )
        )
        yield "places", places
        for place in places:
            yield "place", place
        yield "unresolved", [query for query, place in zip(queries, found) if not place]

    @staticmethod
    async def _resolve_query(
        api_key: SecretStr, query: str, region_code: str
    ) -> dict[str, Any] | None:
        return await resolve_query(api_key, query, region_code)


class GoogleMapsResolveLinksBlock(Block):
    """Find the place behind Google Maps links, including short links."""

    class Input(BlockSchemaInput):
        credentials: GoogleMapsCredentialsInput = GoogleMapsCredentialsField()
        urls: list[str] = SchemaField(
            description=(
                "Google Maps links to resolve, up to 20: google.com/maps links or "
                "maps.app.goo.gl short links"
            ),
            min_length=1,
            max_length=20,
        )
        look_up_place: bool = SchemaField(
            description=(
                "Look up each linked place for its place ID, address and types. "
                "Turn off to only expand and read the links, which is free."
            ),
            default=True,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        results: list[ResolvedMapsLink] = SchemaField(
            description="What each link points at, in the order given"
        )
        result: ResolvedMapsLink = SchemaField(description="Each resolved link")
        failed: list[FailedMapsLink] = SchemaField(
            description="Links that couldn't be resolved, with the reason"
        )

    def __init__(self):
        test_result = ResolvedMapsLink(
            url=_TEST_SHORT_LINK,
            expanded_url=_TEST_EXPANDED_LINK,
            **place_fields(_TEST_PLACE),
        )
        super().__init__(
            id="d25659a9-1d08-4839-8cf5-61c8553185be",
            description=(
                "Find the place a Google Maps link points to, including short "
                "share links: place ID, name, address, coordinates and types. "
                "Works with google.com/maps and maps.app.goo.gl links, up to 20 "
                "at once."
            ),
            categories={BlockCategory.SEARCH},
            input_schema=GoogleMapsResolveLinksBlock.Input,
            output_schema=GoogleMapsResolveLinksBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "urls": [_TEST_SHORT_LINK, _TEST_DIRECTIONS_LINK],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("results", [test_result]),
                ("result", test_result),
                (
                    "failed",
                    [
                        FailedMapsLink(
                            url=_TEST_DIRECTIONS_LINK,
                            reason="This is a directions link, not a link to one place.",
                        )
                    ],
                ),
            ],
            test_mock={
                "_expand_link": lambda url: (
                    _TEST_EXPANDED_LINK if url == _TEST_SHORT_LINK else url
                ),
                "_find_place": lambda api_key, target: _TEST_PLACE,
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            outcomes = await asyncio.gather(
                *(
                    self._resolve_link(
                        credentials.api_key, url, input_data.look_up_place
                    )
                    for url in input_data.urls
                )
            )
        except GoogleMapsError as e:
            raise BlockExecutionError(
                message=str(e), block_name=self.name, block_id=self.id
            ) from e
        lookups = sum(outcome.looked_up for outcome in outcomes)
        self.merge_stats(
            NodeExecutionStats(provider_cost=float(lookups), provider_cost_type="items")
        )
        results = [outcome.result for outcome in outcomes if outcome.result]
        yield "results", results
        for result in results:
            yield "result", result
        yield "failed", [outcome.failure for outcome in outcomes if outcome.failure]

    async def _resolve_link(
        self, api_key: SecretStr, url: str, look_up: bool
    ) -> _LinkOutcome:
        try:
            target = parse_link(await self._expand_link(url))
            place = (
                await self._find_place(api_key, target)
                if look_up and (target.place_id or target.name)
                else None
            )
        except LinkError as e:
            return _LinkOutcome(failure=FailedMapsLink(url=url, reason=str(e)))
        if place:
            return _LinkOutcome(
                result=ResolvedMapsLink(
                    url=url, expanded_url=target.expanded_url, **place_fields(place)
                ),
                looked_up=True,
            )
        if look_up and target.name and not target.place_id:
            near = " at the link's location" if target.latitude is not None else ""
            return _LinkOutcome(
                failure=FailedMapsLink(
                    url=url,
                    reason=f"Google Maps found no place called '{target.name}'{near}.",
                )
            )
        return _LinkOutcome(
            result=ResolvedMapsLink(
                url=url,
                expanded_url=target.expanded_url,
                place_id=target.place_id,
                name=target.name,
                latitude=target.latitude,
                longitude=target.longitude,
            )
        )

    @staticmethod
    async def _expand_link(url: str) -> str:
        return await expand_link(url)

    @staticmethod
    async def _find_place(
        api_key: SecretStr, target: LinkTarget
    ) -> dict[str, Any] | None:
        return await find_link_place(api_key, target)


async def resolve_query(
    api_key: SecretStr, query: str, region_code: str = ""
) -> dict[str, Any] | None:
    """Place details (name, address, types, link) for the best match, or None."""
    if not query:
        return None
    place_id = parse_place_id(query) or await search_place_id(
        api_key, query, region_code=region_code
    )
    if not place_id:
        return None
    try:
        return await get_place(
            api_key, place_id, PLACE_PRO_FIELDS, region_code=region_code
        )
    except GoogleMapsError as e:
        if e.status in (400, 404):
            return None
        raise
