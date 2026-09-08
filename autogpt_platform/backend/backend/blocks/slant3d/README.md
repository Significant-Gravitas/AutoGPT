# Slant3D blocks

These blocks use the [Slant3D v2 API](https://slant3dapi.com/documentation/introduction), updated following the [September 8, 2026 announcement](https://x.com/slant3d/status/2097393736254959819).

## Setup and ordering

1. Connect a Slant3D v2 API key and create a platform in the Slant3D dashboard. Order drafts require a payment method on the Slant3D account, but do not charge it.
2. Supply `platform_id` for orders and file uploads. It can be omitted when the account has exactly one enabled platform.
3. Get filaments and use a filament's `publicId` as `filament_id`. Filters support PLA, PETG, OPM, and color names. Existing color/profile inputs still work when they identify exactly one available filament.
4. Use the Slicer block to upload a public STL URL and estimate its cost. Its `file_id` output can be reused in order items. Order items also accept public `file_url` inputs, uploading and confirming each file before drafting.
5. Estimate Order and Estimate Shipping create uncharged drafts and return `order_id`. Pass that ID to Process Order to charge the account's payment method and submit the draft to production. Create Order combines drafting and processing.

## Compatibility

Existing block IDs and output names are retained. Order IDs are now Slant3D public IDs such as `SLANT_1234567890`. Quantity accepts existing numeric strings and sends a positive integer. `order_number` is stored as `metadata.orderNumber`; the v1 phone and residential-address inputs remain accepted but are not sent in the v2 address contract.

The Filament block retains `filament`, `hexColor`, and `colorTag`, and adds the v2 public ID, material, color, and availability fields. Slicer additionally returns the uploaded file ID. Order listing follows pagination; tracking reads the order's fulfillment information.

New webhook subscriptions require an explicit platform ID and verify Slant3D's timestamped HMAC-SHA256 signature. Each platform has one webhook URL, so use a dedicated platform if another application already has a subscription. Existing v1 subscriptions retain their legacy payload and unsigned-delivery behavior. Reconnect the trigger with a v2 key and platform ID to migrate it. Carrier codes are empty when the provider does not supply them; dummy deliveries do not trigger workflows.

## API contracts and validation

The refreshed [human-readable documentation](https://slant3dapi.com/documentation/introduction) takes precedence where the [OpenAPI specification](https://slant3dapi.com/v2/api/openapi.json) still describes earlier v2 shapes: draft requests put `platformId` inside `customer`, draft prices come from `data.totals`, processing returns the order directly in `data`, and Get Order wraps it in `data.order`.

Run `poetry run pytest backend/blocks/slant3d -q` from the backend directory. Tests mock HTTP boundaries and exercise all block examples, including upload confirmation, pricing, pagination, payment failure, and webhook authentication. They do not place real orders.
