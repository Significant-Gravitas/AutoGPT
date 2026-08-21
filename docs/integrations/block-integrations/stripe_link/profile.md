# Stripe Link Profile
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Stripe Link Get Shipping Address

### What it is
Get the delivery addresses saved on the user's Link wallet, with the default one resolved for you. Use it for any purchase that ships something.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| addresses | Every shipping address on the account, each with an `id`, `is_default` and an `address` object | List[Dict[str, Any]] |
| default_address | The address object marked default, or the first one if none is. Empty when the account has no addresses. | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Stripe Link Get User Info

### What it is
Get the Link account holder's name, email and phone. Use it to fill in a checkout that asks who the buyer is. Pairs with Get Shipping Address for anything physical.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
