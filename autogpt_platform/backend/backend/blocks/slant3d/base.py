from typing import Any, Dict

from backend.data.block import Block
from backend.util.request import requests

from ._api import Color, CustomerDetails, OrderItem, Profile


class Slant3DBlockBase(Block):
    """Base block class for Slant3D API interactions"""

    BASE_URL = "https://www.slant3dapi.com/api"

    def _get_headers(self, api_key: str) -> Dict[str, str]:
        return {"api-key": api_key, "Content-Type": "application/json"}

    def _make_request(self, method: str, endpoint: str, api_key: str, **kwargs) -> Dict:
        url = f"{self.BASE_URL}/{endpoint}"
        response = requests.request(
            method=method, url=url, headers=self._get_headers(api_key), **kwargs
        )

        if not response.ok:
            error_msg = response.json().get("error", "Unknown error")
            raise RuntimeError(f"API request failed: {error_msg}")

        return response.json()

    def _check_valid_color(self, profile: Profile, color: Color, api_key: str) -> str:
        response = self._make_request(
            "GET",
            "filament",
            api_key,
            params={"profile": profile.value, "color": color.value},
        )
        if profile == Profile.PLA:
            color_tag = color.value
        else:
            color_tag = f"{profile.value.lower()}{color.value.capitalize()}"
        valid_tags = [filament["colorTag"] for filament in response["filaments"]]

        if color_tag not in valid_tags:
            raise ValueError(
                f"""Invalid color profile combination {color_tag}.
Valid colors for {profile.value} are:
{','.join([filament['colorTag'].replace(profile.value.lower(), '') for filament in response['filaments'] if filament['profile'] == profile.value])}
"""
            )
        return color_tag

    def _convert_to_color(self, profile: Profile, color: Color, api_key: str) -> str:
        return self._check_valid_color(profile, color, api_key)

    def _format_order_data(
        self,
        customer: CustomerDetails,
        order_number: str,
        items: list[OrderItem],
        api_key: str,
    ) -> list[dict[str, Any]]:
        """Helper function to format order data for API requests"""
        orders = []
        for item in items:
            order_data = {
                "email": customer.email,
                "phone": customer.phone,
                "name": customer.name,
                "orderNumber": order_number,
                "filename": item.file_url,
                "fileURL": item.file_url,
                "bill_to_street_1": customer.address,
                "bill_to_city": customer.city,
                "bill_to_state": customer.state,
                "bill_to_zip": customer.zip,
                "bill_to_country_as_iso": customer.country_iso,
                "bill_to_is_US_residential": str(customer.is_residential).lower(),
                "ship_to_name": customer.name,
                "ship_to_street_1": customer.address,
                "ship_to_city": customer.city,
                "ship_to_state": customer.state,
                "ship_to_zip": customer.zip,
                "ship_to_country_as_iso": customer.country_iso,
                "ship_to_is_US_residential": str(customer.is_residential).lower(),
                "order_item_name": item.file_url,
                "order_quantity": item.quantity,
                "order_image_url": "",
                "order_sku": "NOT_USED",
                "order_item_color": self._convert_to_color(
                    item.profile, item.color, api_key
                ),
                "profile": item.profile.value,
            }
            orders.append(order_data)
        return orders
