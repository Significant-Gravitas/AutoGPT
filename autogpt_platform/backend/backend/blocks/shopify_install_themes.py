import os
import requests
import base64
import hashlib
import json
import urllib
from typing import  List, Dict
import time

# redis
from backend.data import redis
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class ShopifyInstallThemeBlock(Block):
    block_id: str = "f0306d27-f7c2-4b7c-89b2-6b2811048443"

    themes: List[Dict[str, str]] = [
        {
            "description": "A streamlined theme designed to quickly start selling.",
            "id": "gid://shopify/OnlineStoreInstallableThemePreset/294913",
            "name": "Spotlight"
        },
        {
            "description": "A minimalist theme that lets product images take center stage.",
            "id": "gid://shopify/OnlineStoreInstallableThemePreset/100333",
            "name": "Dawn"
        },
        {
            "description": "A refined theme that celebrates craftsmanship.",
            "id": "gid://shopify/OnlineStoreInstallableThemePreset/100341",
            "name": "Craft"
        },
        {
            "description": "A bold theme that elevates product quality and brand storytelling.",
            "id": "gid://shopify/OnlineStoreInstallableThemePreset/131073",
            "name": "Refresh"
        },
        {
            "description": "An energizing theme featuring extensive product detail layouts.",
            "id": "gid://shopify/OnlineStoreInstallableThemePreset/100337",
            "name": "Sense"
        },
        {
            "description": "A stylish theme with special attention to artists and collections.",
            "id": "gid://shopify/OnlineStoreInstallableThemePreset/100343",
            "name": "Studio"
        },
        {
            "description": "A stylish theme designed for makers selling unique pieces",
            "id": "gid://shopify/OnlineStoreInstallableThemePreset/196609",
            "name": "Origin"
        },
        {
            "description": "A distinct, dynamic theme that champions the world of sports.",
            "id": "gid://shopify/OnlineStoreInstallableThemePreset/100349",
            "name": "Ride"
        },
        {
            "description": "An avant-garde theme inspired by independent studios and publishers.",
            "id": "gid://shopify/OnlineStoreInstallableThemePreset/229377",
            "name": "Publisher"
        },
        {
            "description": "Ideal for speciality products and bold branding.",
            "id": "gid://shopify/OnlineStoreInstallableThemePreset/100345",
            "name": "Taste"
        },
        {
            "description": "An eye-catching theme optimized for shopping on the go.",
            "id": "gid://shopify/OnlineStoreInstallableThemePreset/100339",
            "name": "Crave"
        },
        {
            "description": "A striking theme ideal for high-end fashion.",
            "id": "gid://shopify/OnlineStoreInstallableThemePreset/100347",
            "name": "Colorblock"
        },
        {
            "description": "A flexible theme with detail-focused layouts for catalogs of any size",
            "id": "gid://shopify/OnlineStoreInstallableThemePreset/360449",
            "name": "Rise"
        }
    ]

    class Input(BlockSchema):
        shop_name: str = SchemaField(
            description="The name of Shopify shop and subdomain",
        )
        wait_for_complete_seconds: int = SchemaField(
            description="Number of seconds to wait for theme installation to complete",
            default=5,
        )
        user_prompt: str = SchemaField(
            description="User prompt to select theme",
        )

    class Output(BlockSchema):
        shop_name: str = SchemaField(description="The shop that will install a theme")
        shop_preview_url: str = SchemaField(description="The preview url of the theme")
        theme_id: str = SchemaField(description="The theme that was installed")

    def __init__(self):
        self.redis = redis.get_redis()

        super().__init__(
            id=ShopifyInstallThemeBlock.block_id,
            description="This block installs theme a theme on Shopify for user.",
            categories={BlockCategory.SHOPIFY},
            input_schema=ShopifyInstallThemeBlock.Input,
            output_schema=ShopifyInstallThemeBlock.Output,
            test_input=[
                {"shop_name": "3tn-demo"},
            ],
            test_output=[
                ("shop_name", "3tn-demo"),
                ("shop_preview_url", "https://3tn-demo.myshopify.com"),
                ("theme_id", "gid://shopify/OnlineStoreTheme/140466454723"),
            ],
        )


    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        if os.getenv("DEBUG", "false").lower() == "true":
            yield "shop_name", input_data.shop_name
            yield "shop_preview_url", "https://example.com"
            yield "theme_id", "gid://shopify/OnlineStoreTheme/140466454723"
            return

        # TODO: should have a wayt to inject cookie dynamically
        bearer_token = self.generate_session_token(input_data.shop_name)

        auth_code = self.generate_activate_shop_code(input_data.shop_name)
        shop_preview_url = self.activate_shop(input_data.shop_name, auth_code, bearer_token)

        selected_theme = self.select_theme(input_data.user_prompt)
        theme_id = self.install_theme(input_data.shop_name, bearer_token, selected_theme)

        # Delay for the specified amount of time
        time.sleep(input_data.wait_for_complete_seconds)
        yield "shop_name", input_data.shop_name
        yield "shop_preview_url", self.generate_preview_theme_id(shop_preview_url, theme_id)
        yield "theme_id", theme_id
    
    def generate_session_token(self, shop_name: str) -> str:
        url = f"https://admin.shopify.com/api/shopify/{shop_name}?operation=GenerateSessionToken&type=mutation"

         # Load environment variables
        # encoded_cookie = os.getenv("SHOPIFY_INTEGRATION_STORE_COOKIE")
        encoded_cookie = self.redis.get("SHOPIFY_INTEGRATION_STORE_COOKIE")

        if not encoded_cookie:
            raise EnvironmentError("Environment variable SHOPIFY_INTEGRATION_STORE_COOKIE is missing.")
        
        cookie = base64.b64decode(encoded_cookie).decode("utf-8")

        # csrf_token = os.getenv("SHOPIFY_INTEGRATION_STORE_CSRF_TOKEN")

        csrf_token = self.redis.get("SHOPIFY_INTEGRATION_STORE_CSRF_TOKEN")
        
        if not csrf_token:
            raise EnvironmentError("Environment variable 'SHOPIFY_INTEGRATION_STORE_CSRF_TOKEN' is not set.")
        
    
        headers = {
            "accept": "application/json",
            "accept-language": "en,en-US;q=0.9",
            "cache-control": "no-cache",
            "caller-pathname": f"/store/{shop_name}/themes",
            "content-type": "application/json",
            "cookie": cookie,
            "origin": "https://admin.shopify.com",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "referer": f"https://admin.shopify.com/store/{shop_name}/themes",
            "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "target-manifest-route-id": "themes:online-store",
            "target-pathname": f"/store/:storeHandle/themes",
            "target-slice": "onlinestore-section",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "x-csrf-token": csrf_token,
            "x-shopify-web-force-proxy": "1"
        }
        payload = {
            "operationName": "GenerateSessionToken",
            "variables": {
                "appId": "gid://shopify/App/580111"
            },
            "query": "mutation GenerateSessionToken($appId: ID!) { adminGenerateSession(appId: $appId) { session needsMerchantAuthorization __typename } }"
        }

        response = requests.post(url, headers=headers, json=payload)

        # Check for errors
        if response.status_code != 200:
            raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

        data = response.json()["data"]
        if data and data["adminGenerateSession"] and data["adminGenerateSession"]["session"]:
            return data["adminGenerateSession"]["session"]
        else:
            raise Exception(f"Failed to create Shopify store: {data}")

    def generate_activate_shop_code(self, shop_name: str) -> dict[str, str]:
        url = f"https://admin.shopify.com/api/shopify/{shop_name}?operation=GenerateAuthCode&type=mutation"

         # Load environment variables
        # encoded_cookie = os.getenv("SHOPIFY_INTEGRATION_STORE_COOKIE")
        encoded_cookie = self.redis.get("SHOPIFY_INTEGRATION_STORE_COOKIE")

        if not encoded_cookie:
            raise EnvironmentError("Environment variable SHOPIFY_INTEGRATION_STORE_COOKIE is missing.")
        
        cookie = base64.b64decode(encoded_cookie).decode("utf-8")

        #csrf_token = os.getenv("SHOPIFY_INTEGRATION_STORE_CSRF_TOKEN")

        csrf_token = self.redis.get("SHOPIFY_INTEGRATION_STORE_CSRF_TOKEN")

        if not csrf_token:
            raise EnvironmentError("Environment variable 'SHOPIFY_INTEGRATION_STORE_CSRF_TOKEN' is not set.")
        

        headers = {
            "accept": "application/json",
            "accept-language": "en,en-US;q=0.9",
            "cache-control": "no-cache",
            "caller-pathname": f"/store/{shop_name}/themes",
            "content-type": "application/json",
            "cookie": cookie,
            "origin": "https://admin.shopify.com",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "referer": f"https://admin.shopify.com/store/{shop_name}/themes?appLoadId=9daa0964-9347-429b-b7b3-505c2e31a253",
            "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "target-manifest-route-id": "themes:online-store",
            "target-pathname": "/store/:storeHandle/themes",
            "target-slice": "onlinestore-section",
            "user-agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
            "x-csrf-token": csrf_token,
            "x-shopify-web-force-proxy": "1",
        }

        data = {
            "operationName": "GenerateAuthCode",
            "variables": {"appId": "gid://shopify/App/580111"},
            "query": (
                "mutation GenerateAuthCode($appId: ID!) {"
                "  adminGenerateAuthorizationCode(appId: $appId) {"
                "    authorizationCode {"
                "      code"
                "      hmac"
                "      shop"
                "      timestamp"
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
            ),
        }

        response = requests.post(url, headers=headers, json=data)

        raw = response.json()
        data = raw["data"]

        code = data.get("adminGenerateAuthorizationCode", {}).get("authorizationCode", {}).get("code")
        hmac = data.get("adminGenerateAuthorizationCode", {}).get("authorizationCode", {}).get("hmac")
        shop = data.get("adminGenerateAuthorizationCode", {}).get("authorizationCode", {}).get("shop")
        timestamp = data.get("adminGenerateAuthorizationCode", {}).get("authorizationCode", {}).get("timestamp")
        
        if code and hmac and shop and timestamp:
            return {
                "code": code,
                "hmac": hmac,
                "shop": shop,
                "timestamp": timestamp
            }
        else:
            raise Exception(f"Failed to create Shopify store: {raw}")

    def activate_shop(self, shop_name: str, auth_code: dict[str, str],  bearer_token: str) -> str:
        # Base URL
        base_url = "https://online-store-web.shopifyapps.com/admin/online-store/admin/api/unversioned/graphql"

        # Encode the `authCodePayload` dictionary into a URL-safe JSON string
        auth_code_payload = urllib.parse.quote(json.dumps(auth_code))

        # Construct the full URL with query parameters
        full_url = f"{base_url}?authCodePayload={auth_code_payload}&operation=RequestDetails"

        # Headers
        headers = {
            "accept": "application/json",
            "accept-language": "en,en-US;q=0.9",
            "authorization": f"Bearer {bearer_token}",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "origin": "https://online-store-web.shopifyapps.com",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "macOS",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "x-online-store-web": "1",
            "x-requested-with": "XMLHttpRequest",
        }

        # Payload for the POST request
        payload = {
            "operationName": "RequestDetails",
            "variables": {
                "flagNames": [
                    "d6b1323a", "d3e1c4ff", "74d45a1f", "778675f5", "d5340202", "b7299212",
                    "852799de", "2846dc74", "0c10b561", "9b4ca3e9", "00c78352", "e12bbf30",
                    "a219dd0c", "10c98b12", "336502b1", "f71000cb", "ccfc0423", "c1821c61",
                    "fe744fa3", "0d8ae56a", "0eb46c71", "6323254a", "ab2debd1", "e05e5f2c",
                    "6a354d68", "32799051", "120b5fa2", "45817603", "2e506cec", "cbf4ad7c",
                    "4725354a"
                ]
            },
            "query": """
            query RequestDetails($flagNames: [String!]!) {
                shop {
                    id
                    ...EveryRequestShopFragment
                    shopifyEmployeeSession
                    enabledFlags(names: $flagNames)
                    consolidatedSalesTransitionCelebration: experimentAssignment(
                        name: "consolidated_sales_transition_celebration"
                    )
                    developmentShop
                    __typename
                }
                congratulatoryStoreOpenToastDismissed: elementDismissed(
                    handle: "consolidated_sales_transition_celebration_toast"
                )
                onlineStore {
                    ...EveryRequestOnlineStoreFragment
                    __typename
                }
                staffMember {
                    ...EveryRequestStaffMemberFragment
                    __typename
                }
            }

            fragment EveryRequestShopFragment on Shop {
                id
                url
                name
                email
                myshopifyDomain
                ianaTimezone
                timezoneOffsetMinutes
                currencyCode
                countryCode
                chinaCdnEnabled
                developerPreviewName
                starterPlan
                primaryDomain {
                    id
                    host
                    __typename
                }
                plan {
                    name
                    shopifyPlus
                    __typename
                }
                newCustomerAccountUrl: customerAccountNextUrl
                features {
                    usBasedTracking
                    hasGlobalEClassicApp
                    hasGlobalEApp
                    giftCards
                    productProtection
                    checkoutLiquidConfigurationEligible
                    blogContentGenerationEnabled
                    pageContentGenerationEnabled
                    customDataTaxonomyEnabled
                    themePreviewEnabled
                    wooMagicThemeImportEnabled
                    wixMagicThemeImportEnabled
                    wordpressMagicThemeImportEnabled
                    magicThemeImportEnabled
                    __typename
                }
                developerPreview {
                    name
                    __typename
                }
                appPinningExperimentAssignment: experimentAssignment(
                    name: "app_pinning_and_launching"
                )
                appPinningExperimentAssignmentExistingMerchants: experimentAssignment(
                    name: "app_pinning_and_launching_existing_merchants"
                )
                __typename
            }

            fragment EveryRequestOnlineStoreFragment on OnlineStore {
                domains
                urlWithPasswordBypass
                featureSet {
                    themeLimitedPlan
                    __typename
                }
                __typename
            }

            fragment EveryRequestStaffMemberFragment on StaffMember {
                id
                locale
                email
                permissions {
                    userPermissions
                    __typename
                }
                __typename
            }
            """
        }

        # Send the POST request
        response = requests.post(full_url, headers=headers, json=payload)

        raw = response.json()
        data = raw["data"]
        if data and data["onlineStore"] and data["onlineStore"]["urlWithPasswordBypass"]:
            return data["onlineStore"]["urlWithPasswordBypass"]
        else:
            raise Exception(f"Failed to activate Shopify store {shop_name}: {raw} --- \n {bearer_token}")

    def select_theme(self, user_prompt: str) -> Dict[str, str]:
        md5 = hashlib.md5()
        md5.update(user_prompt.encode('utf-8'))
        number = int(md5.hexdigest(), 16)
        selected_theme = self.themes[number % len(self.themes)]
        return selected_theme
    
    def install_theme(self, shop_name: str, bearer_token: str, preset: Dict[str, str]):
        url = "https://online-store-web.shopifyapps.com/admin/online-store/admin/api/unversioned/graphql?operation=FreeThemeInstallLegacy"
        
        headers = {
            "accept": "application/json",
            "accept-language": "en,en-US;q=0.9",
            "authorization": f"Bearer {bearer_token}",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "origin": "https://online-store-web.shopifyapps.com",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "x-online-store-web": "1",
            "x-requested-with": "XMLHttpRequest"
        }
        
        data = {
            "operationName": "FreeThemeInstallLegacy",
            "variables": {
                "publishTheme": True,
                "presetId": preset['id']
            },
            "query": (
                "mutation FreeThemeInstallLegacy($presetId: ID!, $publishTheme: Boolean) {\n"
                "  onlineStoreFreeThemeInstall(presetId: $presetId, publishTheme: $publishTheme) {\n"
                "    newTheme {\n"
                "      ...ThemeInfo\n"
                "      __typename\n"
                "    }\n"
                "    userErrors {\n"
                "      field\n"
                "      message\n"
                "      __typename\n"
                "    }\n"
                "    __typename\n"
                "  }\n"
                "}\n\n"
                "fragment ThemeInfo on OnlineStoreTheme {\n"
                "  id\n"
                "  name\n"
                "  role\n"
                "  matchedTheme {\n"
                "    name\n"
                "    isPurchasable\n"
                "    downloadUrl\n"
                "    __typename\n"
                "  }\n"
                "  parentId\n"
                "  prefix\n"
                "  processing\n"
                "  processingFailed\n"
                "  editedAt\n"
                "  createdAt\n"
                "  mobileScreenshot: screenshot(height: 600, width: 350)\n"
                "  laptopScreenshot: screenshot(height: 900, width: 1160)\n"
                "  mobileScreenshotRedesign: screenshot(height: 900, width: 350)\n"
                "  thumbnailScreenshot: screenshot(\n"
                "    height: 1080\n"
                "    width: 1350\n"
                "    resizeHeight: 144\n"
                "    resizeWidth: 180\n"
                "  )\n"
                "  previewUrl\n"
                "  previewable\n"
                "  editable\n"
                "  themeStoreId\n"
                "  themeStoreUrl\n"
                "  source\n"
                "  supportedFeatures\n"
                "  metadata {\n"
                "    author\n"
                "    version\n"
                "    name\n"
                "    supportUrl: supportUrlV2\n"
                "    supportEmail\n"
                "    __typename\n"
                "  }\n"
                "  features {\n"
                "    onlineStoreV2Compatible\n"
                "    __typename\n"
                "  }\n"
                "  __typename\n"
                "}"
            )
        }
        
        response = requests.post(url, headers=headers, json=data)

        # Check for errors
        if response.status_code != 200:
            raise Exception(f"Request failed with status code {response.status_code}: {response.text} - {bearer_token}")

        raw = response.json()
        data = raw["data"]
        if data and data["onlineStoreFreeThemeInstall"] and data["onlineStoreFreeThemeInstall"]["newTheme"] and data["onlineStoreFreeThemeInstall"]["newTheme"]["id"]:
            return data["onlineStoreFreeThemeInstall"]["newTheme"]["id"]
        else:
            raise Exception(f"Failed to install theme {preset['name']} on Shopify store {shop_name}: {raw} \n --- {preset['id']} --- \n {bearer_token}")

    def generate_preview_theme_id(self, url: str, theme_id: str) -> str:

        # Parse the URL
        parsed_url = urllib.parse.urlparse(url)
        # Parse the query parameters
        query_params = urllib.parse.parse_qs(parsed_url.query)

        # Replace the value of `preview_theme_id`
        parsed_gid = theme_id.split('/')[-1]
        query_params['preview_theme_id'] = [parsed_gid]

        # Reconstruct the query string
        new_query = urllib.parse.urlencode(query_params, doseq=True)

        # Reconstruct the full URL
        updated_url = parsed_url._replace(query=new_query).geturl()

        return updated_url