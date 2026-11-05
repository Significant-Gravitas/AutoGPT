# Printify Catalog
<!-- MANUAL: file_description -->
Blocks for choosing what to print: product types (blueprints), who prints them and the sizes or colours they offer.
<!-- END MANUAL -->

## Printify Get Print Options

### What it is
List who can print a Printify product type and the variants (sizes, colours) they offer, with the ids a new product needs.

### How it works
<!-- MANUAL: how_it_works -->
Lists the print providers for a blueprint, picks the one given (or the first listed) and returns its variants with their ids, ready to pass to Create Printify Product.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| blueprint_id | Product type id from the catalog | int | Yes |
| print_provider_id | Print provider to use; leave 0 for the first one listed | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| print_providers | Every provider that prints this product type | List[PrintifyPrintProvider] |
| print_provider_id | The provider used below | int |
| variants | Sizes and colours this provider offers | List[PrintifyVariant] |
| variant_ids | Every variant id, ready for Create Printify Product | List[int] |

### Possible use case
<!-- MANUAL: use_case -->
Pick the sticker sizes to sell and get their variant ids for a new product.
<!-- END MANUAL -->

---

## Printify Search Catalog

### What it is
Search the Printify catalog for a product type (a blueprint), such as kiss-cut stickers, and get its blueprint id.

### How it works
<!-- MANUAL: how_it_works -->
Reads the Printify catalog of blueprints and keeps those whose title, brand or model contains every word of the query (singular or plural), up to the limit.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Words the product type's name must contain, e.g. 'sticker' | str | Yes |
| limit | Most results to return (1-50) | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| blueprints | Matching product types; pass an id to Get Print Options | List[PrintifyBlueprint] |

### Possible use case
<!-- MANUAL: use_case -->
Look up the blueprint id for kiss-cut or die-cut stickers before creating a product.
<!-- END MANUAL -->

---
