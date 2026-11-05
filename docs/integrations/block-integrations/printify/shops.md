# Printify Shops
<!-- MANUAL: file_description -->
Blocks for finding the Printify shops on an account.
<!-- END MANUAL -->

## Printify List Shops

### What it is
List your Printify shops and the store each one publishes to (Etsy, Shopify, ...). Every other Printify block needs a shop id.

### How it works
<!-- MANUAL: how_it_works -->
Calls Printify's `GET /v1/shops.json` with the connected personal access token and returns every shop with its id, title and the sales channel it publishes to (for example `etsy`). Each shop is also yielded on its own, so a workflow can act on them one at a time.
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| shops | Every shop on the account | List[PrintifyShop] |
| shop | Each shop, one at a time | PrintifyShop |

### Possible use case
<!-- MANUAL: use_case -->
Find the shop id that the upload, product and publish blocks need, or check which store a shop pushes products to before publishing.
<!-- END MANUAL -->

---
