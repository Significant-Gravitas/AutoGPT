import os
import requests
import base64
import time
from backend.data import redis

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class ShopifyInviteStaffBlock(Block):
    block_id: str = "e2f0ed4c-620a-4221-a2a3-c6787a97fa61"

    class Input(BlockSchema):
        shop_name: str = SchemaField(
            description="The name of Shopify shop and subdomain",
        )
        wait_for_complete_seconds: int = SchemaField(
            description="The number of seconds to wait for the invite to complete",
            default=5,
        )
        email: str = SchemaField(
            description="The staff email you want to invite",
        )
        first_name: str = SchemaField(
            description="The staff first name",
        )
        last_name: str = SchemaField(
            description="The staff last name",
        )

    class Output(BlockSchema):
        shop_name: str = SchemaField(description="The shop that invited staff")
        user_id: str = SchemaField(description="The user id of invited staff")
        oauth_url: str = SchemaField(description="The oauth url of to let staff login and authorize")

    def __init__(self):
        oauth_url = os.getenv("SHOPIFY_INTEGRATION_OAUTH_URL")
        if not oauth_url:
            raise EnvironmentError("Environment variable 'SHOPIFY_INTEGRATION_OAUTH_URL' is not set.")
        self.oauth_url = oauth_url
        self.redis = redis.get_redis()

        super().__init__(
            id=ShopifyInviteStaffBlock.block_id,
            description="This block invites a staff to collaborate on a Shopify store.",
            categories={BlockCategory.SHOPIFY},
            input_schema=ShopifyInviteStaffBlock.Input,
            output_schema=ShopifyInviteStaffBlock.Output,
            test_input=[
                {"shop_name": "3tn-demo", "email": "tuan.nguyen930708+1@gmail.com", "first_name": "Tuan", "last_name": "Nguyen"},
            ],
            test_output=[
                ("shop_name", "3tn-demo"),
                ("user_id", "gid://shopify/StaffMember/116287635752"),
                ("oauth_url", self.get_oauth_url("3tn-demo"))
            ],
        )


    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        if os.getenv("DEBUG", "false").lower() == "true":
            yield "shop_name", input_data.shop_name
            yield "user_id", "gid://shopify/StaffMember/116287635752"
            yield "oauth_url",  self.get_oauth_url(input_data.shop_name)
            return

        user_id = self.invite_staff_member(input_data.shop_name, input_data.email, input_data.first_name, input_data.last_name)

        # Delay for the specified amount of time
        time.sleep(input_data.wait_for_complete_seconds)
        yield "shop_name", input_data.shop_name
        yield "user_id", user_id
        yield "oauth_url",  self.get_oauth_url(input_data.shop_name)
    
    def get_oauth_url(self, shop_name: str) -> str: 
        return f"{self.oauth_url}?shop={shop_name}"

    def invite_staff_member(self, shop_name: str, email: str, first_name: str, last_name: str):
        # Load environment variables
        #encoded_cookie = os.getenv("SHOPIFY_INTEGRATION_STORE_COOKIE")

        encoded_cookie = self.redis.get("SHOPIFY_INTEGRATION_STORE_COOKIE")
        if not encoded_cookie:
            raise EnvironmentError("Environment variable SHOPIFY_INTEGRATION_STORE_COOKIE is missing.")
        
        cookie = base64.b64decode(encoded_cookie).decode("utf-8")

        #csrf_token = os.getenv("SHOPIFY_INTEGRATION_STORE_CSRF_TOKEN")
        csrf_token = self.redis.get("SHOPIFY_INTEGRATION_STORE_CSRF_TOKEN")
        if not csrf_token:
            raise EnvironmentError("Environment variable 'SHOPIFY_INTEGRATION_STORE_CSRF_TOKEN' is not set.")
        
        # API endpoint
        url = f"https://admin.shopify.com/api/shopify/{shop_name}?operation=InviteStaffMember&type=mutation"
        
        # Headers
        headers = {
            "accept": "application/json",
            "accept-language": "en,en-US;q=0.9",
            "cache-control": "no-cache",
            "caller-pathname": f"/store/{shop_name}/settings/account/new",
            "content-type": "application/json",
            "cookie": cookie,
            "origin": "https://admin.shopify.com",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "referer": f"https://admin.shopify.com/store/{shop_name}/settings/account/new",
            "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "target-manifest-route-id": "settings-account:invite",
            "target-pathname": "/store/:storeHandle/settings/account/new",
            "target-slice": "account-section",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "x-csrf-token": csrf_token,
            "x-shopify-web-force-proxy": "1"
        }
        
        details =  {
            "email": email,
            "firstName": first_name,
            "lastName": last_name,
            "pin": "",
            "permissions": [
                "APPLICATIONS_BILLING",
                "APPLY_DISCOUNTS_TO_DRAFT_ORDERS",
                "APPLY_DISCOUNTS_TO_ORDERS",
                "AUTHENTICATION_MANAGEMENT",
                "BILLING_CHARGES",
                "BILLING_INVOICES_PAY",
                "BILLING_INVOICES_VIEW",
                "BILLING_PAYMENT_METHODS_MANAGE",
                "BILLING_PAYMENT_METHODS_VIEW",
                "BILLING_SETTINGS",
                "BILLING_SUBSCRIPTIONS",
                "BUY_SHIPPING_LABELS",
                "CANCEL_ORDERS",
                "CAPTURE_PAYMENTS_FOR_ORDERS",
                "COLLABORATOR_REQUEST_MANAGEMENT",
                "COLLABORATOR_REQUEST_SETTINGS",
                "CREATE_AND_EDIT_COMPANIES",
                "CREATE_AND_EDIT_CUSTOMERS",
                "CREATE_AND_EDIT_DRAFT_ORDERS",
                "CREATE_AND_EDIT_GIFT_CARDS",
                "CREATE_FILES",
                "CREATE_STORE_CREDIT_ACCOUNT_TRANSACTIONS",
                "CUSTOMERS",
                "CUSTOM_PIXELS_MANAGEMENT",
                "CUSTOM_PIXELS_VIEW",
                "DASHBOARD",
                "DEACTIVATE_GIFT_CARDS",
                "DELETE_COMPANIES",
                "DELETE_CUSTOMERS",
                "DELETE_DRAFT_ORDERS",
                "DELETE_FILES",
                "DELETE_ORDERS",
                "DELETE_PRODUCTS",
                "DOMAINS",
                "DOMAINS_MANAGEMENT",
                "DRAFT_ORDERS",
                "EDIT_FILES",
                "EDIT_ORDERS",
                "EDIT_PRIVATE_APPS",
                "EDIT_PRODUCT_COST",
                "EDIT_PRODUCT_PRICE",
                "EDIT_THEME_CODE",
                "ENABLE_PRIVATE_APPS",
                "ERASE_CUSTOMER_DATA",
                "EXPORT_CUSTOMERS",
                "EXPORT_DRAFT_ORDERS",
                "EXPORT_GIFT_CARDS",
                "EXPORT_ORDERS",
                "EXPORT_PRODUCTS",
                "FULFILL_AND_SHIP_ORDERS",
                "GIFT_CARDS",
                "LINKS",
                "LOCATIONS",
                "MANAGE_ABANDONED_CHECKOUTS",
                "MANAGE_CHECKOUT_SETTINGS",
                "MANAGE_COMPANY_LOCATION_ASSIGNMENTS",
                "MANAGE_DELIVERY_SETTINGS",
                "MANAGE_INVENTORY",
                "MANAGE_ORDERS_INFORMATION",
                "MANAGE_POLICIES",
                "MANAGE_PRODUCTS",
                "MANAGE_STORE_CREDIT_SETTINGS",
                "MANAGE_TAXES_SETTINGS",
                "MARKETING",
                "MARKETING_SECTION",
                "MARK_DRAFT_ORDERS_AS_PAID",
                "MARK_ORDERS_AS_PAID",
                "MERGE_CUSTOMERS",
                "METAOBJECTS_DELETE",
                "METAOBJECTS_EDIT",
                "METAOBJECTS_VIEW",
                "METAOBJECT_DEFINITIONS_DELETE",
                "METAOBJECT_DEFINITIONS_EDIT",
                "METAOBJECT_DEFINITIONS_VIEW",
                "ORDERS",
                "OVERVIEWS",
                "PAGES",
                "PAYMENT_SETTINGS",
                "PAY_DRAFT_ORDERS_BY_CREDIT_CARD",
                "PAY_DRAFT_ORDERS_BY_VAULTED_CARD",
                "PAY_ORDERS_BY_CREDIT_CARD",
                "PAY_ORDERS_BY_VAULTED_CARD",
                "PREFERENCES",
                "PRODUCTS",
                "REFUND_ORDERS",
                "REPORTS",
                "REQUEST_CUSTOMER_DATA",
                "RETURN_ORDERS",
                "SET_PAYMENT_TERMS_FOR_DRAFT_ORDERS",
                "SET_PAYMENT_TERMS_FOR_ORDERS",
                "SHOPIFY_PAYMENTS_ACCOUNTS",
                "SHOPIFY_PAYMENTS_TRANSFERS",
                "STAFF_API_PERMISSION_MANAGEMENT",
                "STAFF_AUDIT_LOG_VIEW",
                "STAFF_MANAGEMENT",
                "STAFF_MANAGEMENT_ACTIVATION",
                "STAFF_MANAGEMENT_CREATE",
                "STAFF_MANAGEMENT_DELETE",
                "STAFF_MANAGEMENT_UPDATE",
                "THEMES",
                "VIEW_COMPANIES",
                "VIEW_FILES",
                "VIEW_PRIVATE_APPS",
                "VIEW_PRODUCT_COSTS",
                "VIEW_STORE_CREDIT_ACCOUNT_TRANSACTIONS",
                "APPLICATIONS"
            ],
            "appPermissions": [],
            "retailData": {
                "posAccess": False,
                "canInitializePos": False,
                "retailRoleId": None
            }
        }
        # Body
        payload = {
            "operationName": "InviteStaffMember",
            "variables": {"details": details},
            "query": (
                "mutation InviteStaffMember($details: StaffMemberInviteInput!) {"
                "  staffMemberInvite(input: $details) {"
                "    staffMember {"
                "      id"
                "      email"
                "      firstName"
                "      lastName"
                "      name"
                "      retailData {"
                "        posAccess"
                "        retailRole {"
                "          id"
                "          name"
                "          __typename"
                "        }"
                "        __typename"
                "      }"
                "      __typename"
                "    }"
                "    userErrors {"
                "      field"
                "      message"
                "      __typename"
                "    }"
                "    __typename"
                "  }"
                "}"
            )
        }
        
        # Send POST request
        response = requests.post(url, headers=headers, json=payload)
        
        # Check for errors
        if response.status_code != 200:
            raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

        raw = response.json()
        data = raw["data"]
        if data and data["staffMemberInvite"] and data["staffMemberInvite"]["staffMember"] and data["staffMemberInvite"]["staffMember"]["id"]:
            return data["staffMemberInvite"]["staffMember"]["id"]
        else:
            raise Exception(f"Failed to invite staff to store {shop_name}: {raw}")
