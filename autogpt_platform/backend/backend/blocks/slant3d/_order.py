import uuid

from backend.blocks._base import BlockSchemaInput
from backend.data.model import SchemaField

from ._api import (
    TEST_CREDENTIALS_INPUT,
    CustomerDetails,
    OrderItem,
    Slant3DCredentialsField,
    Slant3DCredentialsInput,
)


class OrderInput(BlockSchemaInput):
    credentials: Slant3DCredentialsInput = Slant3DCredentialsField()
    platform_id: str = SchemaField(
        default="",
        description="Slant3D platform ID; may be omitted when the account has one enabled platform",
    )
    order_number: str = SchemaField(
        description="Your custom order reference, stored as orderNumber in Slant3D metadata",
        default_factory=lambda: str(uuid.uuid4()),
    )
    customer: CustomerDetails = SchemaField(description="Customer shipping details")
    items: list[OrderItem] = SchemaField(description="Items to print", min_length=1)


TEST_ORDER_INPUT = {
    "credentials": TEST_CREDENTIALS_INPUT,
    "platform_id": "55555555-5555-4555-8555-555555555555",
    "order_number": "TEST-001",
    "customer": {
        "name": "John Doe",
        "email": "john@example.com",
        "phone": "123-456-7890",
        "address": "123 Test St",
        "city": "Test City",
        "state": "TS",
        "zip": "12345",
    },
    "items": [
        {
            "file_id": "22222222-2222-4222-8222-222222222222",
            "filament_id": "33333333-3333-4333-8333-333333333333",
            "quantity": 1,
        }
    ],
}

TEST_DRAFT = {
    "order": {"publicId": "SLANT_1234567890", "status": "DRAFT"},
    "totals": {"printingCost": 3.75, "deliveryCost": 5.56, "totalCost": 9.31},
}
