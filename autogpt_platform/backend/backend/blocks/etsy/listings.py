from typing import Literal

from backend.blocks.etsy._api import (
    EtsyClient,
    EtsyError,
    EtsyListing,
    EtsyMoney,
    csv,
    flag,
)
from backend.blocks.etsy._config import (
    ETSY_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    EtsyCredentials,
    EtsyCredentialsInput,
    etsy,
)
from backend.sdk import (
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    SchemaField,
)
from backend.util.money import cents_text, price_in_cents

MAX_TAGS = 13
MAX_LISTINGS = 100
# Etsy answers a publish it will not make with a 400 naming what is missing.
PUBLISH_REFUSED = 400
PUBLISH_NEEDS = (
    "To go live, a physical listing needs a photo, a shipping profile and, "
    "unless the shop is in the EU, a return policy. Get My Etsy Shop lists the "
    "shop's shipping profiles and return policies; pass them here as "
    "shipping_profile_id and return_policy_id."
)

ListingState = Literal["active", "inactive", "draft", "sold_out", "expired"]

_TEST_LISTING = EtsyListing(
    listing_id=1500000001,
    title="Sleepy Cat Vinyl Sticker",
    state="draft",
    url="https://www.etsy.com/listing/1500000001/sleepy-cat-vinyl-sticker",
    quantity=999,
    tags=["cat sticker", "kawaii"],
    price=EtsyMoney(amount=399, divisor=100, currency_code="USD"),
)


class EtsyCreateDraftListingBlock(Block):
    """Create an unpublished Etsy listing the seller can review before it goes live."""

    class Input(BlockSchemaInput):
        credentials: EtsyCredentialsInput = etsy.credentials_field(
            description="Your Etsy shop"
        )
        shop_id: int = SchemaField(description="Shop id from Get My Etsy Shop")
        title: str = SchemaField(description="Listing title buyers see")
        description: str = SchemaField(description="Listing description")
        price: float = SchemaField(
            description="Price in your shop's currency, e.g. 3.99"
        )
        quantity: int = SchemaField(
            description="How many are for sale; print on demand can use a large number",
            default=999,
        )
        tags: list[str] = SchemaField(
            description=f"Up to {MAX_TAGS} search tags", default_factory=list
        )
        taxonomy_id: int = SchemaField(
            description="Category id from Search Etsy Categories"
        )
        shipping_profile_id: int = SchemaField(
            description=(
                "Shipping profile from Get My Etsy Shop; a physical listing "
                "needs one before it can be published, not to be drafted"
            ),
            default=0,
        )
        return_policy_id: int = SchemaField(
            description=(
                "Return policy from Get My Etsy Shop; outside the EU a physical "
                "listing needs one to be published"
            ),
            default=0,
        )
        readiness_state_id: int = SchemaField(
            description=(
                "Processing profile from Get My Etsy Shop (how long it takes "
                "to make); 0 leaves it unset, to choose later on Etsy"
            ),
            default=0,
            advanced=True,
        )
        listing_type: Literal["physical", "download"] = SchemaField(
            description="A physical item, or a digital download", default="physical"
        )
        who_made: Literal["i_did", "someone_else", "collective"] = SchemaField(
            description="Who made it, as Etsy asks", default="i_did", advanced=True
        )
        when_made: str = SchemaField(
            description="When it was made, as Etsy asks; print on demand is made_to_order",
            default="made_to_order",
            advanced=True,
        )
        is_supply: bool = SchemaField(
            description="True for craft supplies rather than finished products",
            default=False,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        listing_id: int = SchemaField(description="The new draft listing's id")
        url: str = SchemaField(description="Link to the listing on Etsy")
        listing: EtsyListing = SchemaField(description="The created draft listing")

    def __init__(self):
        super().__init__(
            id="49e07de6-0cbb-43ef-8f7c-fff97e67a406",
            description=(
                "Create a draft Etsy listing with title, description, tags and "
                "price. It stays unpublished until you publish it."
            ),
            categories={BlockCategory.MARKETING},
            input_schema=EtsyCreateDraftListingBlock.Input,
            output_schema=EtsyCreateDraftListingBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "shop_id": 5550123,
                "title": "Sleepy Cat Vinyl Sticker",
                "description": "A matte vinyl sticker of a very sleepy cat.",
                "price": 3.99,
                "tags": ["cat sticker", "kawaii"],
                "taxonomy_id": 2078,
                "shipping_profile_id": 2001,
            },
            test_output=[
                ("listing_id", _TEST_LISTING.listing_id),
                ("url", _TEST_LISTING.url),
                ("listing", _TEST_LISTING),
            ],
            test_mock={"create": lambda credentials, shop_id, form: _TEST_LISTING},
            effect=BlockEffect.EXTERNAL_JUDGED,
            disabled=not ETSY_OAUTH_IS_CONFIGURED,
        )

    @staticmethod
    async def create(
        credentials: EtsyCredentials, shop_id: int, form: dict[str, str]
    ) -> EtsyListing:
        return await EtsyClient(credentials).create_draft_listing(shop_id, form)

    async def run(
        self, input_data: Input, *, credentials: EtsyCredentials, **kwargs
    ) -> BlockOutput:
        listing = await self.create(
            credentials, input_data.shop_id, draft_listing_form(input_data)
        )
        yield "listing_id", listing.listing_id
        yield "url", listing.url
        yield "listing", listing


def draft_listing_form(data: EtsyCreateDraftListingBlock.Input) -> dict[str, str]:
    """The createDraftListing form, checked for what Etsy would refuse."""
    cents = price_in_cents(data.price)
    if len(data.tags) > MAX_TAGS:
        raise ValueError(f"Etsy allows at most {MAX_TAGS} tags per listing.")
    form = {
        "title": data.title,
        "description": data.description,
        "price": cents_text(cents),
        "quantity": str(data.quantity),
        "taxonomy_id": str(data.taxonomy_id),
        "who_made": data.who_made,
        "when_made": data.when_made,
        "is_supply": flag(data.is_supply),
        "type": data.listing_type,
    }
    if data.tags:
        form["tags"] = csv(data.tags)
    if data.shipping_profile_id:
        form["shipping_profile_id"] = str(data.shipping_profile_id)
    if data.return_policy_id:
        form["return_policy_id"] = str(data.return_policy_id)
    if data.readiness_state_id:
        form["readiness_state_id"] = str(data.readiness_state_id)
    return form


class EtsyListListingsBlock(Block):
    """List a shop's listings in one state, e.g. everything that is live."""

    class Input(BlockSchemaInput):
        credentials: EtsyCredentialsInput = etsy.credentials_field(
            description="Your Etsy shop"
        )
        shop_id: int = SchemaField(description="Shop id from Get My Etsy Shop")
        state: ListingState = SchemaField(
            description="Which listings: active (live), draft, inactive, sold_out or expired",
            default="active",
        )
        limit: int = SchemaField(
            description=f"Most listings to return (1-{MAX_LISTINGS})", default=25
        )

    class Output(BlockSchemaOutput):
        listings: list[EtsyListing] = SchemaField(
            description="The shop's listings in that state", default_factory=list
        )

    def __init__(self):
        super().__init__(
            id="f258abd8-abe7-4b2b-996c-55367ce83b73",
            description=(
                "List your Etsy listings by state, for example everything that "
                "is live, with title, price, tags and link."
            ),
            categories={BlockCategory.MARKETING, BlockCategory.DATA},
            input_schema=EtsyListListingsBlock.Input,
            output_schema=EtsyListListingsBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "shop_id": 5550123},
            test_output=[("listings", [_TEST_LISTING])],
            test_mock={
                "list_listings": lambda credentials, shop_id, state, limit: [
                    _TEST_LISTING
                ]
            },
            effect=BlockEffect.READ,
            disabled=not ETSY_OAUTH_IS_CONFIGURED,
        )

    @staticmethod
    async def list_listings(
        credentials: EtsyCredentials, shop_id: int, state: str, limit: int
    ) -> list[EtsyListing]:
        return await EtsyClient(credentials).list_listings(shop_id, state, limit)

    async def run(
        self, input_data: Input, *, credentials: EtsyCredentials, **kwargs
    ) -> BlockOutput:
        limit = max(1, min(input_data.limit, MAX_LISTINGS))
        yield "listings", await self.list_listings(
            credentials, input_data.shop_id, input_data.state, limit
        )


class EtsyUpdateListingBlock(Block):
    """Edit a listing's text and tags, or publish or unpublish it."""

    class Input(BlockSchemaInput):
        credentials: EtsyCredentialsInput = etsy.credentials_field(
            description="Your Etsy shop"
        )
        shop_id: int = SchemaField(description="Shop id from Get My Etsy Shop")
        listing_id: int = SchemaField(description="Listing to change")
        title: str = SchemaField(description="New title; empty keeps it", default="")
        description: str = SchemaField(
            description="New description; empty keeps it", default=""
        )
        tags: list[str] = SchemaField(
            description=f"New tags (up to {MAX_TAGS}); empty keeps them",
            default_factory=list,
        )
        state: Literal["", "active", "inactive"] = SchemaField(
            description="'active' publishes the listing, 'inactive' hides it; empty keeps it",
            default="",
        )
        shipping_profile_id: int = SchemaField(
            description=(
                "Shipping profile from Get My Etsy Shop; a physical listing "
                "needs one to be published. 0 keeps it"
            ),
            default=0,
        )
        return_policy_id: int = SchemaField(
            description=(
                "Return policy from Get My Etsy Shop; outside the EU a physical "
                "listing needs one to be published. 0 keeps it"
            ),
            default=0,
        )

    class Output(BlockSchemaOutput):
        listing: EtsyListing = SchemaField(description="The listing after the change")
        url: str = SchemaField(description="Link to the listing on Etsy")

    def __init__(self):
        super().__init__(
            id="6e2ba5d6-f985-4fb9-b13a-4ca8af7dfa89",
            description=(
                "Update an Etsy listing's title, description or tags, or publish "
                "a draft (state 'active') so buyers can see it."
            ),
            categories={BlockCategory.MARKETING},
            input_schema=EtsyUpdateListingBlock.Input,
            output_schema=EtsyUpdateListingBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "shop_id": 5550123,
                "listing_id": _TEST_LISTING.listing_id,
                "state": "active",
            },
            test_output=[("listing", _TEST_LISTING), ("url", _TEST_LISTING.url)],
            test_mock={
                "update": lambda credentials, shop_id, listing_id, form: _TEST_LISTING
            },
            effect=BlockEffect.EXTERNAL,
            disabled=not ETSY_OAUTH_IS_CONFIGURED,
        )

    @staticmethod
    async def update(
        credentials: EtsyCredentials,
        shop_id: int,
        listing_id: int,
        form: dict[str, str],
    ) -> EtsyListing:
        return await EtsyClient(credentials).update_listing(shop_id, listing_id, form)

    async def run(
        self, input_data: Input, *, credentials: EtsyCredentials, **kwargs
    ) -> BlockOutput:
        form = update_listing_form(input_data)
        try:
            listing = await self.update(
                credentials, input_data.shop_id, input_data.listing_id, form
            )
        except EtsyError as e:
            if form.get("state") != "active" or e.status_code != PUBLISH_REFUSED:
                raise
            raise EtsyError(f"{e} {PUBLISH_NEEDS}", e.status_code) from e
        yield "listing", listing
        yield "url", listing.url


def update_listing_form(data: EtsyUpdateListingBlock.Input) -> dict[str, str]:
    if len(data.tags) > MAX_TAGS:
        raise ValueError(f"Etsy allows at most {MAX_TAGS} tags per listing.")
    form = {
        "title": data.title,
        "description": data.description,
        "tags": csv(data.tags),
        "state": data.state,
        "shipping_profile_id": _id(data.shipping_profile_id),
        "return_policy_id": _id(data.return_policy_id),
    }
    changes = {key: value for key, value in form.items() if value}
    if not changes:
        raise ValueError(
            "Nothing to change: give a title, description, tags, state, "
            "shipping profile or return policy."
        )
    return changes


def _id(value: int) -> str:
    return str(value) if value else ""
