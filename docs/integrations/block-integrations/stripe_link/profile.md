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
returns an empty object when the account has no addresses.
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| addresses | Every shipping address on the account, each with an `id`, `is_default` and an `address` object | List[Dict[str, Any]] |
| default_address | The address object marked default, or the first one if none is. Empty when the account has no addresses. | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
Use `default_address` to prefill the delivery fields for a physical purchase,
or present `addresses` to the user when they should choose among several saved
destinations. Confirm that the selected address satisfies the merchant's
country and postal-code requirements before submitting the order.
<!-- END MANUAL -->

---

## Stripe Link Get User Info

### What it is
Get the Link account holder's name, email and phone. Use it to fill in a checkout that asks who the buyer is. Pairs with Get Shipping Address for anything physical.

### How it works
<!-- MANUAL: how_it_works -->
The block makes an authenticated `GET /userinfo` request to Link and maps the
returned name, email, and phone fields directly to block outputs. Missing keys
become empty strings, but an explicit null from Link is passed through rather
than normalized. A failed Link request is reported through `error`.
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
Prefill the buyer-name and contact fields for a checkout, then combine them
with Get Shipping Address for an order that needs delivery. Treat empty outputs
as missing profile data and ask for the required value instead of submitting an
incomplete checkout.
<!-- END MANUAL -->

---
