# Printify Products
<!-- MANUAL: file_description -->
Blocks for creating, publishing and listing products in a Printify shop.
<!-- END MANUAL -->

## Printify Create Product

### What it is
Create a Printify product (unpublished) that prints your design on a product type, with title, description, tags and price.

### How it works
<!-- MANUAL: how_it_works -->
Builds Printify's create-product request: every chosen variant enabled at one retail price (converted to cents), and the design placed centred at full scale on the chosen print position (`front` by default). The product is created unpublished.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| shop_id | Shop id from List Printify Shops | int | Yes |
| title | Product title buyers see | str | Yes |
| description | Product description (HTML allowed) | str | Yes |
| tags | Search tags for the listing | List[str] | No |
| blueprint_id | Product type id from the catalog | int | Yes |
| print_provider_id | Print provider id | int | Yes |
| variant_ids | Variants (sizes, colours) to sell, from Get Print Options | List[int] | Yes |
| price | Retail price per item in your shop's currency, e.g. 3.99 | float | Yes |
| image_id | Design id from Upload Printify Image | str | Yes |
| print_position | Where the design is printed; stickers and most items use 'front' | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| product_id | The new product's id | str |
| product | The created product | PrintifyProduct |

### Possible use case
<!-- MANUAL: use_case -->
Turn each uploaded sticker design into a Printify product with an SEO title, description and tags, ready for review.
<!-- END MANUAL -->

---

## Printify List Products

### What it is
List the products in a Printify shop, with whether each is visible in the store.

### How it works
<!-- MANUAL: how_it_works -->
Lists the products in a shop (up to 50 per call) with whether each is visible in the store and whether a publish is still in progress.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| shop_id | Shop id from List Printify Shops | int | Yes |
| limit | Most products to return (1-50) | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| products | Products in the shop | List[PrintifyProduct] |

### Possible use case
<!-- MANUAL: use_case -->
Check which stickers are already in the shop before creating duplicates, or report what was added this week.
<!-- END MANUAL -->

---

## Printify Publish Product

### What it is
Publish a Printify product to the store connected to its shop (for example your Etsy shop), making it live for buyers.

### How it works
<!-- MANUAL: how_it_works -->
Calls Printify's publish endpoint with every part (title, description, images, variants, tags and shipping) so the product is pushed to the store connected to its shop, such as an Etsy shop.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| shop_id | Shop id from List Printify Shops | int | Yes |
| product_id | Product id to publish | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| product_id | The product that is publishing | str |

### Possible use case
<!-- MANUAL: use_case -->
Publish a reviewed sticker product so it goes live on the connected Etsy store.
<!-- END MANUAL -->

---
