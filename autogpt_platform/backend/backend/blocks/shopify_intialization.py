import os
import requests
import base64
import time

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class ShopifyInitializeBlock(Block):
    block_id: str = "0f173175-176b-4648-92c5-e6eb001f1bfc"

    class Input(BlockSchema):
        shop_name: str = SchemaField(
            description="The name of Shopify shop and subdomain",
        )
        wait_for_complete_seconds: int = SchemaField(
            description="The number of seconds to wait for the store to be created",
            default=15,
        )

    class Output(BlockSchema):
        shop_name: str = SchemaField(description="The shop name on Shopify")
        shop_url: str = SchemaField(description="The shop url on Shopify")

    def __init__(self):
        super().__init__(
            id=ShopifyInitializeBlock.block_id,
            description="This block intializes a Shopify store.",
            categories={BlockCategory.SHOPIFY},
            input_schema=ShopifyInitializeBlock.Input,
            output_schema=ShopifyInitializeBlock.Output,
            test_input=[
                {"shop_name": "3tn-demo"},
            ],
            test_output=[
                ("shop_name", "3tn-demo"),
                ("shop_id", "68905730226")
            ],
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        if os.getenv("DEBUG", "false").lower() == "true":
            yield "shop_name", input_data.shop_name
            yield "shop_url", "https://example.com"
            return

        shop_url = self.create_shopify_store(input_data.shop_name)

        # Delay for the specified amount of time
        time.sleep(input_data.wait_for_complete_seconds)
        yield "shop_name", input_data.shop_name
        yield "shop_url", shop_url

    def create_shopify_store(self, shop_name):
        # Retrieve environment variables
        encoded_cookie = os.getenv("SHOPIFY_INTEGRATION_PARTNER_COOKIE")
        if not encoded_cookie:
            raise EnvironmentError("Environment variable 'SHOPIFY_INTEGRATION_PARTNER_COOKIE' is not set.")
        
        cookie = base64.b64decode(encoded_cookie).decode('utf-8')

        csrf_token = os.getenv("SHOPIFY_INTEGRATION_PARTNER_CSRF_TOKEN")
        if not csrf_token:
            raise EnvironmentError("Environment variable 'SHOPIFY_INTEGRATION_PARTNER_CSRF_TOKEN' is not set.")
        
        partner_id = os.getenv("SHOPIFY_INTEGRATION_PARTNER_ID")
        if not partner_id:
            raise EnvironmentError("Environment variable 'SHOPIFY_INTEGRATION_PARTNER_ID' is not set.")
        
        country_code = os.getenv("SHOPIFY_INTEGRATION_COUNTRY_CODE", "VN")

        # Define the endpoint and headers
        url = f"https://partners.shopify.com/{partner_id}/api/graphql"
        headers = {
            "accept": "*/*, application/json",
            "accept-language": "en,en-US;q=0.9",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "cookie": cookie,
            "origin": "https://partners.shopify.com",
            "pragma": "no-cache",
            "referer": f"https://partners.shopify.com/{partner_id}/stores/new?store_type=client_store",
            "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "x-csrf-token": csrf_token,
        }

        # Define the payload
        payload = {
            "operationName": "ShopCreateMutation",
            "variables": {
                "input": {
                    "storeType": "DEV_STORE",
                    "storeName": shop_name,
                    "address": {
                        "countryCode": country_code
                    },
                    "subdomain": shop_name,
                    "brickAndMortarPresence": False,
                    "merchantSellingThroughAnotherPlatform": True
                }
            },
            "query": """
                mutation ShopCreateMutation($input: ShopCreateInput!) {
                shopCreate(input: $input) {
                    redirectUrl
                    userErrors {
                    field
                    message
                    __typename
                    }
                    __typename
                }
                }
            """
        }

        # Make the POST request
        response = requests.post(url, json=payload, headers=headers)

        # Check for errors
        if response.status_code != 200:
            raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

        raw = response.json()
        data = raw["data"]
        if data and data["shopCreate"] and data["shopCreate"]["redirectUrl"]:
            return data["shopCreate"]["redirectUrl"]
        else:
            raise Exception(f"Failed to create Shopify store {shop_name}: {raw}")
