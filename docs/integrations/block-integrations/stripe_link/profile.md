# Stripe Link Profile
<!-- MANUAL: file_description -->
These read-only blocks retrieve the connected Link account's saved contact and
shipping information. They can supply checkout fields, but they do not create
or approve a spend request.
<!-- END MANUAL -->

## Stripe Link Get Shipping Address

### What it is
Get the delivery addresses saved on the user's Link wallet, with the default one resolved for you. Use it for any purchase that ships something.

### How it works
<!-- MANUAL: how_it_works -->
The block makes an authenticated `GET /shipping_addresses` request to Link and
returns the complete list under `addresses`. For `default_address`, it selects
the address marked as default, falls back to the first saved address, and
returns an empty object when the account has no addresses. If the authenticated
Link request fails, the block emits `error` instead of address outputs.
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| addresses | Every shipping address on the account, each with an `id`, `is_default` and an `address` object | List[Dict[str, Any]] |
| default_address | The address object marked default, or the first one if none is. Empty when the account has no addresses. | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Default Delivery**: Prefill a physical purchase with the wallet's `default_address`.

**Address Choice**: Present `addresses` when the user should choose among several saved destinations.

**Fulfillment Validation**: Confirm that the selected address meets the merchant's country and postal-code requirements.
<!-- END MANUAL -->

---

## Stripe Link Get User Info

### What it is
Get the Link account holder's name, email and phone. Use it to fill in a checkout that asks who the buyer is. Pairs with Get Shipping Address for anything physical.

### How it works
<!-- MANUAL: how_it_works -->
The block makes an authenticated `GET /userinfo` request to Link and maps the
returned name, first name, last name, email, and phone fields directly to block
outputs. Missing keys and explicit null values become empty strings so they
satisfy the output schema. A failed Link request is reported through `error`.
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| name | Full name on the Link account | str |
| first_name | Given name | str |
| last_name | Family name | str |
| email | Email on the Link account | str |
| phone | Phone number in E.164 format, e.g. +15551234567 | str |

### Possible use case
<!-- MANUAL: use_case -->
**Checkout Prefill**: Fill buyer-name and contact fields from the connected Link profile.

**Physical Orders**: Combine contact details with Get Shipping Address for a delivered purchase.

**Missing Data Recovery**: Ask the user for any required value returned as an empty string.
<!-- END MANUAL -->

---
