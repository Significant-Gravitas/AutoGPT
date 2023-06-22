from pydantic import BaseModel, create_model, ValidationError, root_validator
from typing import List, Type, Any, Union


class PolyBaseModel(BaseModel):
    """A base model that can be subclassed by other models"""
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v) -> Any:
        """Validate that the input can be parsed into one of the subclasses"""
        if not isinstance(v, dict):
            raise ValueError("Must be a dict")

        for subclass in cls.__subclasses__():
            if "type" in v.keys():
                type_value = v.get('type')
                if type_value == subclass.__name__:
                    del v["type"]
                    return subclass(**v)
            else:
                try:
                    return subclass(**v)
                except ValidationError:
                    pass
        raise ValueError("None of the subclasses match")
    
class BaseAddress(PolyBaseModel):
    street: str
    city: str
    zip_code: str

class USAAddress(BaseAddress):
    state: str

class UKAddress(BaseAddress):
    county: str

class CanadaAddress(BaseAddress):
    state: str

class User(BaseModel):
    name: str
    age: int
    addresses: List[BaseAddress]



json_data_usa_addresses = """
{
    "name": "John Doe",
    "age": 30,
    "addresses": [
        {
            "type": "USAAddress",
            "street": "123 Main St",
            "city": "New York",
            "zip_code": "10001",
            "state": "NY"
        },
        {
        "type": "USAAddress",
            "street": "456 Broadway St",
            "city": "New York",
            "zip_code": "10002",
            "state": "NY"
        }
    ]
}
"""

json_data_uk_addresses = """
{
    "name": "John Doe",
    "age": 30,
    "addresses": [
        {
            "street": "123 Main St",
            "city": "New York",
            "zip_code": "10001",
            "county": "NY"
        },
        {
            "street": "456 Broadway St",
            "city": "New York",
            "zip_code": "10002",
            "county": "NY"
        }
    ]
}
"""


import json

# Parse the json data
use_user = User.parse_raw(json_data_usa_addresses)
uk_user = User.parse_raw(json_data_uk_addresses)

# Print the user
print("Example of a user with USA addresses")
print(use_user)
print("Example of a user with UK addresses")
print(uk_user)

