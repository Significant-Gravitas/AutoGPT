import os
import json
import shopify
from typing import Any

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import BlockSecret, SchemaField, SecretField


class ShopifyProductCreateBlock(Block):
    block_id: str = "ee90f1a3-25b6-4495-ba1c-dee65bb55685"
    api_version: str = "2025-01"

    class Input(BlockSchema):
        shop_name: str = SchemaField(
            description="The name of Shopify shop and subdomain",
        )
        products: list[dict[str, Any]] = SchemaField(description="List of products to create")
        admin_api_key: str = SchemaField(description="Shopify app API key for Admin API")

    class Output(BlockSchema):
        shop_name: str = SchemaField(description="The shop that invited staff")
        products: list[dict[str, Any]] = SchemaField(description="List of products created")

    def __init__(self):
        super().__init__(
            id=ShopifyProductCreateBlock.block_id,
            description="This block create products on Shopify.",
            categories={BlockCategory.SHOPIFY},
            input_schema=ShopifyProductCreateBlock.Input,
            output_schema=ShopifyProductCreateBlock.Output,
            test_input=[
                {"shop_name": "3tn-demo", "admin_api_key": "admin_api_key"},
            ],
            test_output=[
                ("shop_name", "3tn-demo"),
                ("products", []),
            ],
        )


    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        if os.getenv("DEBUG", "false").lower() == "true":
            yield "shop_name", input_data.shop_name
            yield "products", []
            return
        
        shop_url = f"https://{input_data.shop_name}.myshopify.com"
        session = shopify.Session(
            shop_url, 
            self.api_version,
            input_data.admin_api_key
        )
        shopify.ShopifyResource.activate_session(session)

        location = self.get_location()
        if not location:
            raise ValueError("Could not find location")
        
        products = self.parse_products(input_data.products)
        for item in products:
            product = self.create_product(item)
            item["id"] = product["id"]

            existing_variants = self.get_variants(product["id"])
            self.update_product_variant_price(product["id"], existing_variants)
            item["variants_created"] = self.create_variants(item, product, existing_variants)

            inventory_items = self.get_inventory_items(product["id"])
            item["inventory_items_tracked"] = self.track_inventory_items(inventory_items)
            item["inventory_items_changes"] = self.update_inventory_items(inventory_items, location["id"], item["quantity_maps"])

        yield "shop_name", input_data.shop_name
        yield "products", products

    def get_location(self) -> dict[str, str]:
        query = "query { locations(first: 1) { nodes { id name address { address1 city province country zip } isActive fulfillsOnlineOrders } } }"
        response = shopify.GraphQL().execute(query)

        raw = json.loads(response)
        data = raw["data"]

        nodes = data.get("locations", {}).get("nodes", [])
        if not nodes:
            raise ValueError("No locations found")

        location = dict(
            id=nodes[0]["id"],
            name=nodes[0]["name"],
        )
        return location

    def parse_products(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        products = {}
        
        for item in items: 
            title = item.get("Title", "")
            if not title: continue

            product = {
                "title": item.get("Title", ""),
                "description": item.get("Description", ""),
                "variants": [],
                "options": {},
                "quantity_maps": {}
            } if title not in products else products[title]

            variant = {"price": item.get("Price", 0), "title": "", "values": []}
            variant_title = []
            for key, value in item.items():
                if key.startswith("Variant"):
                    variant_title.append(value)

                    option = key.split(":")[-1].strip()
                    
                    variant["values"].append({"name": value, "option": option})

                    if option not in product["options"]:
                        product["options"][option] = [value]
                    elif value not in product["options"][option]:
                        product["options"][option].append(value)

            variant["title"] = " / ".join(variant_title)
            product["variants"].append(variant)
            product["quantity_maps"][variant["title"]] = int(item.get("Quantity", 0))
            products[title] = product

        for product in products.values():
            options = product["options"]
            product["options"] = []
            for key, values in options.items():
                product["options"].append({"name": key, "values": [{"name": value} for value in values]})
            
        return list(products.values())
    
    def create_product(self, item: dict[str, Any]) -> dict[str, str]:
        query = "mutation productCreate($input: ProductInput!) {  productCreate(    input: $input  ) {    product {      id      title      options {        id        name        values      }    }    userErrors {      field      message    }  }}"

        params = {
            "input": {
                "title": item.get("title", ""),
                "descriptionHtml": item.get("description", ""),
                "productOptions": item.get("options", []),
            }
        }
        response = shopify.GraphQL().execute(query, params)
        raw = json.loads(response)
        data = raw["data"]

        errors = data.get("productCreate", {}).get("userErrors", {})
        if errors:
            raise ValueError("Could not create product because of errors", errors)

        product = data.get("productCreate", {}).get("product", {})
        if not product:
            raise ValueError("Could not create product")

        return dict(product)

    def get_variants(self, product_id: str) -> list[dict[str, str]]:
        query = "query getProduct($id: ID!) { product(id: $id) { id title description options { id name values } variants(first: 100) { edges { cursor node { id title price } } } } }"
        params = {
            "id": product_id,
        }
        response = shopify.GraphQL().execute(query, params)

        raw = json.loads(response)
        data = raw["data"]

        edges = data.get("product", {}).get("variants", {}).get("edges", [])
        if not edges:
            raise ValueError("Could not get product variants")
        
        variants = []
        for edge in edges:
            variants.append(dict(edge["node"]))

        return variants
    
    def create_variants(
        self, 
        item: dict[str, Any], 
        product: dict[str, Any], 
        existing_variants: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        existing_variant_title = [v["title"] for v in existing_variants]

        query = "mutation createProductVariants($productId: ID!, $variants: [ProductVariantsBulkInput!]!) { productVariantsBulkCreate(productId: $productId, variants: $variants) { productVariants { id title price selectedOptions { name value } } userErrors { field message } } }"

        params = {
            "productId": product["id"],
            "variants": []
        }

        for item_variant in item["variants"]:
            if item_variant["title"] not in existing_variant_title:
                variant = {
                    "price": item_variant["price"],
                    "optionValues": []
                }
                
                for option in item_variant["values"]:
                    variant["optionValues"].append({
                        "optionName": option["option"], 
                        "name": option["name"]
                    })

                params["variants"].append(variant)
                    
        # no new variants
        if not params["variants"]:
            return list()

        response = shopify.GraphQL().execute(query, params)

        raw = json.loads(response)
        data = raw["data"]

        errors = data.get("productVariantsBulkCreate", {}).get("userErrors", {})
        if errors:
            raise ValueError("Could not create product variants because of errors", errors, params, existing_variants)

        variants = data.get("productVariantsBulkCreate", {}).get("productVariants", [])
        if not variants:
            raise ValueError("Could not create product variants", response)

        return variants

    def update_product_variant_price(self, product_id: str, variants: list[dict[str, str]]) -> list[dict[str, str]]:
        query = "mutation updateProductVariantsPrice($productId: ID!, $variants: [ProductVariantsBulkInput!]!) { productVariantsBulkUpdate(productId: $productId, variants: $variants) { productVariants { id title price } userErrors { field message } } }"
        params = {
            "productId": product_id,
            "variants": [{"id": variant["id"], "price": 19.99} for variant in variants]
        }
        response = shopify.GraphQL().execute(query, params)

        raw = json.loads(response)
        data = raw["data"]

        errors = data.get("productVariantsBulkUpdate", {}).get("userErrors", {})
        if errors:
            raise ValueError("Could not update product variants price because of errors", errors)

        updated = data.get("productVariantsBulkUpdate", {}).get("productVariants", {})

        return [{"id": variant["id"], "title": variant["title"], "price": variant["price"]} for variant in updated if "id" in variant]

    def get_inventory_items(self, product_id: str) -> list[dict[str, str]]:
        query = "query getProductInventoryItems ($query: String) { productVariants(first: 100, query: $query) {  nodes {    inventoryItem {      id  variant { id title }  }   }} }"
        params = {
            "query": f"product_id:{product_id.split('/')[-1]}",
        }
        response = shopify.GraphQL().execute(query, params)

        raw = json.loads(response)
        data = raw["data"]

        nodes = data.get("productVariants", {}).get("nodes", [])
        if not nodes:
            raise ValueError("Could not get product inventory items")
        
        items = []
        for node in nodes:
            items.append(dict(id= node["inventoryItem"]["id"], variant= dict(node["inventoryItem"]["variant"])))

        return items

    def track_inventory_items(self, inventory_items: list[dict[str, str]]) -> dict[str, bool]:
        queries = []
        params_input = []
        params = dict()
        for index, item in enumerate(inventory_items):
            param_name = "id_"+str(index)
            params_input.append("$" + param_name+ ":ID!")
            params[param_name] = item["id"]

            item_query = "inventoryItemUpdate"+str(index)+": inventoryItemUpdate(id: $"+param_name+", input: { tracked: true }) { inventoryItem { id tracked } userErrors { field message } }"
            queries.append(item_query)

        query = "mutation trackInventoryItems(" +  ",".join(params_input) + ") { " + " \n ".join(queries) + " }"
        response = shopify.GraphQL().execute(query, params)
        
        raw = json.loads(response)
        data = raw["data"]

        tracks = dict()
        for key, value in data.items():
            item = value.get("inventoryItem", {})
            if item and item.get("id", "") != "" and item.get("tracked", False):
                tracks[item["id"]] = item["tracked"]

        return tracks
    
    def update_inventory_items(self, inventory_items: list[dict[str, str]], localtion_id: str, quantity_maps: dict[str, int]) -> list[str]:
        queries = []
        params_input = []
        params = dict()
        for index, item in enumerate(inventory_items):
            param_name = "input_"+str(index)
            params_input.append("$" + param_name+ ":InventoryAdjustQuantitiesInput!")
            params[param_name] = {"reason": "correction", "name": "available", "changes": [{"delta": quantity_maps.get(item["variant"]["title"], 0), "inventoryItemId": item["id"], "locationId": localtion_id}]}

            item_query = "inventoryAdjustQuantities"+str(index)+": inventoryAdjustQuantities(input: $"+str(param_name)+") { inventoryAdjustmentGroup { id createdAt } userErrors { field message } }"
            queries.append(item_query)

        query = "mutation adjustInventory(" +  ",".join(params_input) + ") { " + " \n ".join(queries) + " }"
        response = shopify.GraphQL().execute(query, params)

        raw = json.loads(response)
        data = raw["data"]

        changes = []
        for key, value in data.items():
            item = value.get("inventoryAdjustmentGroup", {})
            if item and item.get("id", "") != "":
                changes.append(item.get("id", ""))

        return changes


