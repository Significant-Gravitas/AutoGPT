import os
import json
import shopify

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import BlockSecret, SchemaField, SecretField


class ShopifyProductCreateBlock(Block):
    block_id: str = "ee90f1a3-25b6-4495-ba1c-dee65bb55685"
    api_version: str = "2025-01"

    class Input(BlockSchema):
        shop_name: str = SchemaField(
            description="The name of Shopify shop and subdomain",
        )
        api_key: BlockSecret = SecretField(key="api_key",value="api_key")

    class Output(BlockSchema):
        shop_name: str = SchemaField(description="The shop that invited staff")
        products: list[dict[str, str]] = SchemaField(description="List of products created")

    def __init__(self):
        super().__init__(
            id=ShopifyProductCreateBlock.block_id,
            description="This block create products on Shopify.",
            categories={BlockCategory.SHOPIFY},
            input_schema=ShopifyProductCreateBlock.Input,
            output_schema=ShopifyProductCreateBlock.Output,
            test_input=[
                {"shop_name": "3tn-demo", "api_key": "api_key"},
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
       
        with shopify.Session.temp(shop_url, self.api_version, input_data.api_key.get_secret_value()):
            session = shopify.Session(
                f"https://{input_data.shop_name}.myshopify.com", 
                "2025-01",
                self.api_key
            )
            shopify.ShopifyResource.activate_session(session)

            location = self.get_location()
            product = self.create_product()
            items = self.get_inventory_items(product["id"])
            print("------------------------------------------------------------")
            print(items)
            print("------------------------------------------------------------")

            yield "shop_name", input_data.shop_name
            yield "products", []

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

    def create_product(self) -> dict[str, str]:
        query = "mutation productCreate($input: ProductInput!) {  productCreate(    input: $input  ) {    product {      id      title      options {        id        name        values      }    }    userErrors {      field      message    }  }}"
        params = {
            "input": {
                "title": "My Cool Socks",
                "productOptions": [
                {
                    "name": "Color",
                    "values": [
                    {
                        "name": "Red"
                    },
                    {
                        "name": "Green"
                    },
                    {
                        "name": "Blue"
                    }
                    ]
                },
                {
                    "name": "Size",
                    "values": [
                    {
                        "name": "Small"
                    },
                    {
                        "name": "Medium"
                    },
                    {
                        "name": "Large"
                    }
                    ]
                }
                ]
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
        query = "query getProduct($id: ID!) { product(id: $id) { id title description options { id name values } variants(first: 100) { edges { cursor node { id title price sku } } } } }"
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

    def get_inventory_items(self, product_id: str) -> list[dict[str, str]]:
        query = "query getProductInventoryItems ($query: String) { productVariants(first: 100, query: $query) {  nodes {    inventoryItem {      id  variant { id }  }   }} }"
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
            items.append(dict(id= node["inventoryItem"]["id"], variant_id= node["inventoryItem"]["variant"]["id"]))

        return items
