from pydantic import SecretStr
import redis
from enum import Enum
from typing import List, Optional, Literal, cast

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import (
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
    UserPasswordCredentials,
)
from backend.integrations.providers import ProviderName

RedisCredentials = UserPasswordCredentials
RedisCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.REDIS],
    Literal["user_password"],
]

def RedisCredentialsField() -> RedisCredentialsInput:
    """Creates a Redis credentials input on a block."""
    return CredentialsField(
        description="Redis connection credentials",
    )

TEST_REDIS_CREDENTIALS = UserPasswordCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="redis",
    username=SecretStr("mock-redis-username"),
    password=SecretStr("mock-redis-password"),
    title="Mock Redis credentials",
)

TEST_REDIS_CREDENTIALS_INPUT = {
    "provider": TEST_REDIS_CREDENTIALS.provider,
    "id": TEST_REDIS_CREDENTIALS.id,
    "type": TEST_REDIS_CREDENTIALS.type,
    "title": TEST_REDIS_CREDENTIALS.title,
}

class ListDirection(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"

class SetAction(str, Enum):
    ADD = "ADD"
    REMOVE = "REMOVE"

class SetQueryAction(str, Enum):
    GET_ALL = "GET_ALL"  # Corresponds to SMEMBERS
    IS_MEMBER = "IS_MEMBER" # Corresponds to SISMEMBER

class SetCondition(str, Enum):
    NX = "NX" # Only set the key if it does not already exist.
    XX = "XX" # Only set the key if it already exist.


class RedisGetBlock(Block):
    """Retrieves the value stored at a specific key."""
    class Input(BlockSchema):
        credentials: RedisCredentialsInput = RedisCredentialsField()
        host: str = SchemaField(description="Redis server host address")
        port: int = SchemaField(description="Redis server port", default=6379,advanced=False)
        ssl: bool = SchemaField(description="Whether to use SSL for the connection", default=False)
        key: str = SchemaField(description="The key whose value to retrieve.")

    class Output(BlockSchema):
        success: bool = SchemaField(description="Operation succeeded.")
        value: Optional[str] = SchemaField(description="The value stored at the key, or None if the key doesn't exist.")
        error:str = SchemaField(description="Error message if operation failed.")

    def __init__(self):
        super().__init__(
            id="a00438b3-64e8-4b4b-9e45-911231791314",
            description="Retrieves the value stored at a specific key in Redis.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=RedisGetBlock.Input,
            output_schema=RedisGetBlock.Output,
            test_credentials=TEST_REDIS_CREDENTIALS,
            test_input={
                "credentials": TEST_REDIS_CREDENTIALS_INPUT,
                "host": "localhost",
                "port": 6379,
                "ssl": False,
                "key": "my_test_key"
            },
            test_output=[
                ("success", True),
                ("value", "my_test_value")
            ],
            test_mock={
                "run": lambda *args, **kwargs: [
                    ("success", True),
                    ("value", "my_test_value")
                ]
            },
        )

    def run(
        self, input_data: Input, *, credentials: RedisCredentials, **kwargs
    ) -> BlockOutput:
        try:
            with redis.Redis(
                host=input_data.host,
                port=input_data.port,
                username=credentials.username.get_secret_value() if credentials.username else "default",
                password=credentials.password.get_secret_value() if credentials.password else None,
                ssl=input_data.ssl,
                decode_responses=True # Decode from bytes to str automatically
            ) as r:
                value = r.get(input_data.key)

                yield "success", True
                yield "value", value
        except Exception as e:
            yield "success", False
            yield "error", str(e)

class RedisSetBlock(Block):
    """Stores or updates a value for a key, optionally setting an expiry time or conditions."""
    class Input(BlockSchema):
        credentials: RedisCredentialsInput = RedisCredentialsField()
        host: str = SchemaField(description="Redis server host address")
        port: int = SchemaField(description="Redis server port", default=6379, advanced=False)
        ssl: bool = SchemaField(description="Whether to use SSL for the connection", default=False)
        key: str = SchemaField(description="The key to set.")
        value: str = SchemaField(description="The value to store.")
        expiration_ms: Optional[int] = SchemaField(
            default=None,
            description="Optional expiration time in milliseconds (PX).",
            advanced=True
        )
        condition: Optional[SetCondition] = SchemaField(
            default=None,
            description="Set condition: NX (Not Exists) or XX (Exists).",
            advanced=True
        )

    class Output(BlockSchema):
        success: bool = SchemaField(description="True if the SET operation was successful.")
        key_was_set: Optional[bool] = SchemaField(description="True if the key was actually set (especially relevant with NX/XX). Can be None if command fails early.")
        error: str = SchemaField(description="Error message if operation failed.")

    def __init__(self):
        super().__init__(
            id="b78a6a2d-ff2e-4e5a-a9a5-92e5dafbccf5",
            description="Stores or updates a value for a key in Redis, with optional expiry and conditions.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=RedisSetBlock.Input,
            output_schema=RedisSetBlock.Output,
            test_credentials=TEST_REDIS_CREDENTIALS,
            test_input={
                "credentials": TEST_REDIS_CREDENTIALS_INPUT,
                "host": "localhost",
                "port": 6379,
                "ssl": False,
                "key": "my_set_key",
                "value": "some data",
                "expiration_ms": 60000, # 1 minute
                "condition": None,
            },
            test_output=[
                ("success", True),
                ("key_was_set", True)
            ],
             test_mock={
                "run": lambda *args, **kwargs: [
                    ("success", True),
                    ("key_was_set", True)
                ]
            },
        )

    def run(
        self, input_data: Input, *, credentials: RedisCredentials, **kwargs
    ) -> BlockOutput:
        try:
            with redis.Redis(
                host=input_data.host,
                port=input_data.port,
                username=credentials.username.get_secret_value() if credentials.username else "default",
                password=credentials.password.get_secret_value() if credentials.password else None,
                ssl=input_data.ssl,
                decode_responses=True
            ) as r:
                # Prepare arguments for set command
                set_args = {
                    'name': input_data.key,
                    'value': input_data.value,
                    'px': input_data.expiration_ms,
                    'nx': input_data.condition == SetCondition.NX if input_data.condition else None,
                    'xx': input_data.condition == SetCondition.XX if input_data.condition else None,
                }
                # Remove None values as redis-py expects keyword args to be present or absent
                set_args_filtered = {k: v for k, v in set_args.items() if v is not None}

                result = r.set(**set_args_filtered)

                yield "success", True
                # SET returns True if successful, None if NX/XX condition not met.
                yield "key_was_set", bool(result)

        except Exception as e:
            yield "success", False
            yield "key_was_set", None
            yield "error", str(e)

class RedisDeleteBlock(Block):
    """Removes one or more specified keys and their associated values."""
    class Input(BlockSchema):
        credentials: RedisCredentialsInput = RedisCredentialsField()
        host: str = SchemaField(description="Redis server host address")
        port: int = SchemaField(description="Redis server port", default=6379, advanced=False)
        ssl: bool = SchemaField(description="Whether to use SSL for the connection", default=False)
        keys: list[str] = SchemaField(description="The key(s) to delete.")

    class Output(BlockSchema):
        success: bool = SchemaField(description="True if the DELETE operation was successful.")
        deleted_count: int = SchemaField(description="Number of keys that were actually deleted.")
        error: str = SchemaField(description="Error message if operation failed.")

    def __init__(self):
        super().__init__(
            id="c89b3a6d-ff2e-4e5a-a9a5-92e5dafbccf5",
            description="Removes one or more specified keys and their associated values from Redis.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=RedisDeleteBlock.Input,
            output_schema=RedisDeleteBlock.Output,
            test_credentials=TEST_REDIS_CREDENTIALS,
            test_input={
                "credentials": TEST_REDIS_CREDENTIALS_INPUT,
                "host": "localhost",
                "port": 6379,
                "ssl": False,
                "keys": ["my_key1", "my_key2"]
            },
            test_output=[
                ("success", True),
                ("deleted_count", 2)
            ],
            test_mock={
                "run": lambda *args, **kwargs: [
                    ("success", True),
                    ("deleted_count", 2)
                ]
            },
        )

    def run(
        self, input_data: Input, *, credentials: RedisCredentials, **kwargs
    ) -> BlockOutput:
        try:
            with redis.Redis(
                host=input_data.host,
                port=input_data.port,
                username=credentials.username.get_secret_value() if credentials.username else "default",
                password=credentials.password.get_secret_value() if credentials.password else None,
                ssl=input_data.ssl,
                decode_responses=True
            ) as r:
                # Delete the specified keys
                deleted_count = r.delete(*input_data.keys)

                yield "success", True
                yield "deleted_count", deleted_count

        except Exception as e:
            yield "success", False
            yield "deleted_count", 0
            yield "error", str(e)

class RedisExistsBlock(Block):
    """Checks if one or more specified keys exist in the database."""
    class Input(BlockSchema):
        credentials: RedisCredentialsInput = RedisCredentialsField()
        host: str = SchemaField(description="Redis server host address")
        port: int = SchemaField(description="Redis server port", default=6379, advanced=False)
        ssl: bool = SchemaField(description="Whether to use SSL for the connection", default=False)
        keys: list[str] = SchemaField(description="The key(s) to check for existence.")

    class Output(BlockSchema):
        success: bool = SchemaField(description="True if the EXISTS operation was successful.")
        count: int = SchemaField(description="Number of keys that exist in the database.")
        error: str = SchemaField(description="Error message if operation failed.")

    def __init__(self):
        super().__init__(
            id="d7e4f59b-d1c7-4a13-b9f2-08a62c7f0e31",
            description="Checks if one or more specified keys exist in Redis.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=RedisExistsBlock.Input,
            output_schema=RedisExistsBlock.Output,
            test_credentials=TEST_REDIS_CREDENTIALS,
            test_input={
                "credentials": TEST_REDIS_CREDENTIALS_INPUT,
                "host": "localhost",
                "port": 6379,
                "ssl": False,
                "keys": ["my_key1", "my_key2"]
            },
            test_output=[
                ("success", True),
                ("count", 1)
            ],
            test_mock={
                "run": lambda *args, **kwargs: [
                    ("success", True),
                    ("count", 1)
                ]
            },
        )

    def run(
        self, input_data: Input, *, credentials: RedisCredentials, **kwargs
    ) -> BlockOutput:
        try:
            with redis.Redis(
                host=input_data.host,
                port=input_data.port,
                username=credentials.username.get_secret_value() if credentials.username else "default",
                password=credentials.password.get_secret_value() if credentials.password else None,
                ssl=input_data.ssl,
                decode_responses=True
            ) as r:
                # Check if the specified keys exist
                count = r.exists(*input_data.keys)

                yield "success", True
                yield "count", count

        except Exception as e:
            yield "success", False
            yield "count", 0
            yield "error", str(e)

class RedisAtomicCounterBlock(Block):
    """Atomically increases or decreases the integer value stored at a key."""
    class Input(BlockSchema):
        credentials: RedisCredentialsInput = RedisCredentialsField()
        host: str = SchemaField(description="Redis server host address")
        port: int = SchemaField(description="Redis server port", default=6379, advanced=False)
        ssl: bool = SchemaField(description="Whether to use SSL for the connection", default=False)
        key: str = SchemaField(description="The key storing the counter value.")
        increment: int = SchemaField(description="Amount to increment (positive) or decrement (negative).", default=1)
        initial_value: Optional[int] = SchemaField(
            description="Initial value if key doesn't exist yet.",
            default=0,
            advanced=True
        )

    class Output(BlockSchema):
        success: bool = SchemaField(description="True if the operation was successful.")
        new_value: int = SchemaField(description="The new value after the increment/decrement operation.")
        error: str = SchemaField(description="Error message if operation failed.")

    def __init__(self):
        super().__init__(
            id="e2f7a6b3-91c8-4d5a-b8e3-72e4d9f1c8d7",
            description="Atomically increases or decreases the integer value stored at a key in Redis.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=RedisAtomicCounterBlock.Input,
            output_schema=RedisAtomicCounterBlock.Output,
            test_credentials=TEST_REDIS_CREDENTIALS,
            test_input={
                "credentials": TEST_REDIS_CREDENTIALS_INPUT,
                "host": "localhost",
                "port": 6379,
                "ssl": False,
                "key": "my_counter",
                "increment": 5,
                "initial_value": 0
            },
            test_output=[
                ("success", True),
                ("new_value", 5)
            ],
            test_mock={
                "run": lambda *args, **kwargs: [
                    ("success", True),
                    ("new_value", 5)
                ]
            },
        )

    def run(
        self, input_data: Input, *, credentials: RedisCredentials, **kwargs
    ) -> BlockOutput:
        try:
            with redis.Redis(
                host=input_data.host,
                port=input_data.port,
                username=credentials.username.get_secret_value() if credentials.username else "default",
                password=credentials.password.get_secret_value() if credentials.password else None,
                ssl=input_data.ssl,
                decode_responses=True
            ) as r:
                if not r.exists(input_data.key) and input_data.initial_value != 0:
                    r.set(input_data.key, str(input_data.initial_value))
                    if input_data.increment == 0:
                        new_value = input_data.initial_value
                    else:
                        new_value = r.incrby(input_data.key, input_data.increment)
                else:
                    new_value = r.incrby(input_data.key, input_data.increment)

                yield "success", True
                yield "new_value", new_value

        except Exception as e:
            yield "success", False
            yield "new_value", 0
            yield "error", str(e)

class RedisInfoBlock(Block):
    """Retrieves information and statistics about the Redis server instance."""
    class Input(BlockSchema):
        credentials: RedisCredentialsInput = RedisCredentialsField()
        host: str = SchemaField(description="Redis server host address")
        port: int = SchemaField(description="Redis server port", default=6379, advanced=False)
        ssl: bool = SchemaField(description="Whether to use SSL for the connection", default=False)
        section: Optional[str] = SchemaField(
            description="Optional section of information to retrieve (e.g., 'server', 'clients', 'memory'). If not provided, all sections are returned.",
            default=None
        )

    class Output(BlockSchema):
        success: bool = SchemaField(description="True if the operation was successful.")
        info: dict = SchemaField(description="Dictionary containing server information and statistics.")
        error: str = SchemaField(description="Error message if operation failed.")

    def __init__(self):
        super().__init__(
            id="f9a3b582-7e4d-48c1-b6a9-dc5e78f91a34",
            description="Retrieves information and statistics about the Redis server instance.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=RedisInfoBlock.Input,
            output_schema=RedisInfoBlock.Output,
            test_credentials=TEST_REDIS_CREDENTIALS,
            test_input={
                "credentials": TEST_REDIS_CREDENTIALS_INPUT,
                "host": "localhost",
                "port": 6379,
                "ssl": False,
                "section": None
            },
            test_output=[
                ("success", True),
                ("info", {"redis_version": "6.2.6", "uptime_in_seconds": "3600"})
            ],
            test_mock={
                "run": lambda *args, **kwargs: [
                    ("success", True),
                    ("info", {"redis_version": "6.2.6", "uptime_in_seconds": "3600"})
                ]
            },
        )

    def run(
        self, input_data: Input, *, credentials: RedisCredentials, **kwargs
    ) -> BlockOutput:
        try:
            with redis.Redis(
                host=input_data.host,
                port=input_data.port,
                username=credentials.username.get_secret_value() if credentials.username else "default",
                password=credentials.password.get_secret_value() if credentials.password else None,
                ssl=input_data.ssl,
                decode_responses=True
            ) as r:

                info = r.info(section=input_data.section)

                yield "success", True
                yield "info", info

        except Exception as e:
            yield "success", False
            yield "info", {}
            yield "error", str(e)

class RedisListPushBlock(Block):
    """Adds one or more elements to the beginning or end of a list."""
    class Input(BlockSchema):
        credentials: RedisCredentialsInput = RedisCredentialsField()
        host: str = SchemaField(description="Redis server host address")
        port: int = SchemaField(description="Redis server port", default=6379, advanced=False)
        ssl: bool = SchemaField(description="Whether to use SSL for the connection", default=False)
        key: str = SchemaField(description="The key of the list to push to.")
        values: list[str] = SchemaField(description="The value(s) to push to the list.")
        direction: ListDirection = SchemaField(
            description="Direction to push: LEFT (beginning) or RIGHT (end).",
            default=ListDirection.RIGHT
        )

    class Output(BlockSchema):
        success: bool = SchemaField(description="True if the operation was successful.")
        new_length: int = SchemaField(description="The new length of the list after the push operation.")
        error: str = SchemaField(description="Error message if operation failed.")

    def __init__(self):
        super().__init__(
            id="f4b7c8d9-a1b2-4c3d-9e8f-7a6b5c4d3e2f",
            description="Adds one or more elements to the beginning or end of a list in Redis.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=RedisListPushBlock.Input,
            output_schema=RedisListPushBlock.Output,
            test_credentials=TEST_REDIS_CREDENTIALS,
            test_input={
                "credentials": TEST_REDIS_CREDENTIALS_INPUT,
                "host": "localhost",
                "port": 6379,
                "ssl": False,
                "key": "my_list",
                "values": ["value1", "value2"],
                "direction": ListDirection.RIGHT
            },
            test_output=[
                ("success", True),
                ("new_length", 2)
            ],
            test_mock={
                "run": lambda *args, **kwargs: [
                    ("success", True),
                    ("new_length", 2)
                ]
            },
        )

    def run(
        self, input_data: Input, *, credentials: RedisCredentials, **kwargs
    ) -> BlockOutput:
        try:
            with redis.Redis(
                host=input_data.host,
                port=input_data.port,
                username=credentials.username.get_secret_value() if credentials.username else "default",
                password=credentials.password.get_secret_value() if credentials.password else None,
                ssl=input_data.ssl,
                decode_responses=True
            ) as r:
                # Choose push method based on direction
                if input_data.direction == ListDirection.LEFT:
                    new_length = r.lpush(input_data.key, *input_data.values)
                else:  # ListDirection.RIGHT
                    new_length = r.rpush(input_data.key, *input_data.values)

                yield "success", True
                yield "new_length", new_length

        except Exception as e:
            yield "success", False
            yield "new_length", 0
            yield "error", str(e)

class RedisListPopBlock(Block):
    """Removes and returns an element from the beginning or end of a list, optionally waiting if empty."""
    class Input(BlockSchema):
        credentials: RedisCredentialsInput = RedisCredentialsField()
        host: str = SchemaField(description="Redis server host address")
        port: int = SchemaField(description="Redis server port", default=6379, advanced=False)
        ssl: bool = SchemaField(description="Whether to use SSL for the connection", default=False)
        key: str = SchemaField(description="The key of the list to pop from.")
        direction: ListDirection = SchemaField(
            description="Direction to pop from: LEFT (beginning) or RIGHT (end).",
            default=ListDirection.LEFT
        )
        wait_ms: Optional[int] = SchemaField(
            description="Time to wait in milliseconds if the list is empty (0 means no wait).",
            default=0,
            advanced=True
        )

    class Output(BlockSchema):
        success: bool = SchemaField(description="True if the operation was successful.")
        value: Optional[str] = SchemaField(description="The popped element, or None if list is empty and not waiting.")
        error: str = SchemaField(description="Error message if operation failed.")

    def __init__(self):
        super().__init__(
            id="a12b3c4d-e5f6-7g8h-9i0j-k1l2m3n4o5p6",
            description="Removes and returns an element from the beginning or end of a list in Redis, optionally waiting if empty.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=RedisListPopBlock.Input,
            output_schema=RedisListPopBlock.Output,
            test_credentials=TEST_REDIS_CREDENTIALS,
            test_input={
                "credentials": TEST_REDIS_CREDENTIALS_INPUT,
                "host": "localhost",
                "port": 6379,
                "ssl": False,
                "key": "my_list",
                "direction": ListDirection.LEFT,
                "wait_ms": 0
            },
            test_output=[
                ("success", True),
                ("value", "item1")
            ],
            test_mock={
                "run": lambda *args, **kwargs: [
                    ("success", True),
                    ("value", "item1")
                ]
            },
        )

    def run(
        self, input_data: Input, *, credentials: RedisCredentials, **kwargs
    ) -> BlockOutput:
        try:
            with redis.Redis(
                host=input_data.host,
                port=input_data.port,
                username=credentials.username.get_secret_value() if credentials.username else "default",
                password=credentials.password.get_secret_value() if credentials.password else None,
                ssl=input_data.ssl,
                decode_responses=True
            ) as r:
                if input_data.wait_ms and input_data.wait_ms > 0:
                    if input_data.direction == ListDirection.LEFT:
                        result = r.blpop([input_data.key], timeout=int(input_data.wait_ms/1000))
                    else:
                        result = r.brpop([input_data.key], timeout=int(input_data.wait_ms/1000))

                    value = result[1] if result and isinstance(result, (list, tuple)) else None
                else:
                    if input_data.direction == ListDirection.LEFT:
                        value = r.lpop(input_data.key)
                    else:
                        value = r.rpop(input_data.key)

                yield "success", True
                yield "value", value

        except Exception as e:
            yield "success", False
            yield "value", None
            yield "error", str(e)

class RedisListRangeBlock(Block):
    """Retrieves a specified range of elements from a list without removing them."""
    class Input(BlockSchema):
        credentials: RedisCredentialsInput = RedisCredentialsField()
        host: str = SchemaField(description="Redis server host address")
        port: int = SchemaField(description="Redis server port", default=6379, advanced=False)
        ssl: bool = SchemaField(description="Whether to use SSL for the connection", default=False)
        key: str = SchemaField(description="The key of the list to retrieve elements from.")
        start: int = SchemaField(description="Start index (inclusive, 0-based). Negative indices count from the end of the list.", default=0)
        end: int = SchemaField(description="End index (inclusive, 0-based). Negative indices count from the end of the list.", default=-1)

    class Output(BlockSchema):
        success: bool = SchemaField(description="True if the operation was successful.")
        elements: list[str] = SchemaField(description="List of elements retrieved from the specified range.")
        error: str = SchemaField(description="Error message if operation failed.")

    def __init__(self):
        super().__init__(
            id="b23c4d5e-f6g7-8h9i-j0k1-l2m3n4o5p6q7",
            description="Retrieves a specified range of elements from a list in Redis without removing them.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=RedisListRangeBlock.Input,
            output_schema=RedisListRangeBlock.Output,
            test_credentials=TEST_REDIS_CREDENTIALS,
            test_input={
                "credentials": TEST_REDIS_CREDENTIALS_INPUT,
                "host": "localhost",
                "port": 6379,
                "ssl": False,
                "key": "my_list",
                "start": 0,
                "end": -1
            },
            test_output=[
                ("success", True),
                ("elements", ["item1", "item2", "item3"])
            ],
            test_mock={
                "run": lambda *args, **kwargs: [
                    ("success", True),
                    ("elements", ["item1", "item2", "item3"])
                ]
            },
        )

    def run(
        self, input_data: Input, *, credentials: RedisCredentials, **kwargs
    ) -> BlockOutput:
        try:
            with redis.Redis(
                host=input_data.host,
                port=input_data.port,
                username=credentials.username.get_secret_value() if credentials.username else "default",
                password=credentials.password.get_secret_value() if credentials.password else None,
                ssl=input_data.ssl,
                decode_responses=True
            ) as r:
                elements = r.lrange(input_data.key, input_data.start, input_data.end)

                yield "success", True
                yield "elements", elements

        except Exception as e:
            yield "success", False
            yield "elements", []
            yield "error", str(e)

class RedisHashSetBlock(Block):
    """Sets or updates one or multiple field-value pairs within a hash."""
    class Input(BlockSchema):
        credentials: RedisCredentialsInput = RedisCredentialsField()
        host: str = SchemaField(description="Redis server host address")
        port: int = SchemaField(description="Redis server port", default=6379, advanced=False)
        ssl: bool = SchemaField(description="Whether to use SSL for the connection", default=False)
        key: str = SchemaField(description="The key of the hash.")
        fields: dict[str, str] = SchemaField(description="Field-value pairs to set in the hash.")
        nx: bool = SchemaField(
            description="Only set fields that do not already exist.",
            default=False,
            advanced=True
        )

    class Output(BlockSchema):
        success: bool = SchemaField(description="True if the operation was successful.")
        updated_count: int = SchemaField(description="Number of fields that were added or updated.")
        error: str = SchemaField(description="Error message if operation failed.")

    def __init__(self):
        super().__init__(
            id="c3d4e5f6-g7h8-i9j0-k1l2-m3n4o5p6q7r8",
            description="Sets or updates one or multiple field-value pairs within a hash in Redis.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=RedisHashSetBlock.Input,
            output_schema=RedisHashSetBlock.Output,
            test_credentials=TEST_REDIS_CREDENTIALS,
            test_input={
                "credentials": TEST_REDIS_CREDENTIALS_INPUT,
                "host": "localhost",
                "port": 6379,
                "ssl": False,
                "key": "my_hash",
                "fields": {"field1": "value1", "field2": "value2"},
                "nx": False
            },
            test_output=[
                ("success", True),
                ("updated_count", 2)
            ],
            test_mock={
                "run": lambda *args, **kwargs: [
                    ("success", True),
                    ("updated_count", 2)
                ]
            },
        )

    def run(
        self, input_data: Input, *, credentials: RedisCredentials, **kwargs
    ) -> BlockOutput:
        try:
            with redis.Redis(
                host=input_data.host,
                port=input_data.port,
                username=credentials.username.get_secret_value() if credentials.username else "default",
                password=credentials.password.get_secret_value() if credentials.password else None,
                ssl=input_data.ssl,
                decode_responses=True
            ) as r:
                if input_data.nx:
                    # When nx=True, we need to use hsetnx which sets only if field doesn't exist
                    # We need to call it for each field individually
                    updated_count = 0
                    for field, value in input_data.fields.items():
                        if r.hsetnx(input_data.key, field, value):
                            updated_count += 1
                else:
                    # Standard hset can handle multiple field-value pairs at once
                    updated_count = r.hset(input_data.key, mapping=input_data.fields)

                yield "success", True
                yield "updated_count", updated_count

        except Exception as e:
            yield "success", False
            yield "updated_count", 0
            yield "error", str(e)

class RedisHashGetBlock(Block):
    """Retrieves the value(s) associated with one or more specified fields within a hash."""
    class Input(BlockSchema):
        credentials: RedisCredentialsInput = RedisCredentialsField()
        host: str = SchemaField(description="Redis server host address")
        port: int = SchemaField(description="Redis server port", default=6379, advanced=False)
        ssl: bool = SchemaField(description="Whether to use SSL for the connection", default=False)
        key: str = SchemaField(description="The key of the hash.")
        fields: list[str] = SchemaField(description="Field names to retrieve values for.")
        get_all: bool = SchemaField(
            description="If true, retrieves all fields and values in the hash (ignores 'fields' parameter).",
            default=False
        )

    class Output(BlockSchema):
        success: bool = SchemaField(description="True if the operation was successful.")
        values: dict[str, str] = SchemaField(description="Dictionary of field-value pairs retrieved from the hash.")
        error: str = SchemaField(description="Error message if operation failed.")

    def __init__(self):
        super().__init__(
            id="d4e5f6g7-h8i9-j0k1-l2m3-n4o5p6q7r8s9",
            description="Retrieves the value(s) associated with one or more specified fields within a hash in Redis.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=RedisHashGetBlock.Input,
            output_schema=RedisHashGetBlock.Output,
            test_credentials=TEST_REDIS_CREDENTIALS,
            test_input={
                "credentials": TEST_REDIS_CREDENTIALS_INPUT,
                "host": "localhost",
                "port": 6379,
                "ssl": False,
                "key": "my_hash",
                "fields": ["field1", "field2"],
                "get_all": False
            },
            test_output=[
                ("success", True),
                ("values", {"field1": "value1", "field2": "value2"})
            ],
            test_mock={
                "run": lambda *args, **kwargs: [
                    ("success", True),
                    ("values", {"field1": "value1", "field2": "value2"})
                ]
            },
        )

    def run(
        self, input_data: Input, *, credentials: RedisCredentials, **kwargs
    ) -> BlockOutput:
        try:
            with redis.Redis(
                host=input_data.host,
                port=input_data.port,
                username=credentials.username.get_secret_value() if credentials.username else "default",
                password=credentials.password.get_secret_value() if credentials.password else None,
                ssl=input_data.ssl,
                decode_responses=True
            ) as r:
                values = {}

                if input_data.get_all:
                    # Get all fields and values in the hash
                    values = r.hgetall(input_data.key)
                else:
                    if len(input_data.fields) == 1:
                        value = r.hget(input_data.key, input_data.fields[0])
                        if value is not None:
                            values[input_data.fields[0]] = value
                    else:
                        field_values_raw = r.hmget(input_data.key, input_data.fields)
                        field_values = cast(List[Optional[str]], field_values_raw)
                        for field, value in zip(input_data.fields, field_values):
                            if value is not None:
                                values[field] = value

                yield "success", True
                yield "values", values

        except Exception as e:
            yield "success", False
            yield "values", {}
            yield "error", str(e)

class RedisHashGetAllBlock(Block):
    """Retrieves all field-value pairs stored within a specified hash."""
    class Input(BlockSchema):
        credentials: RedisCredentialsInput = RedisCredentialsField()
        host: str = SchemaField(description="Redis server host address")
        port: int = SchemaField(description="Redis server port", default=6379, advanced=False)
        ssl: bool = SchemaField(description="Whether to use SSL for the connection", default=False)
        key: str = SchemaField(description="The key of the hash to retrieve all fields from.")

    class Output(BlockSchema):
        success: bool = SchemaField(description="True if the operation was successful.")
        values: dict[str, str] = SchemaField(description="Dictionary of all field-value pairs in the hash.")
        error: str = SchemaField(description="Error message if operation failed.")

    def __init__(self):
        super().__init__(
            id="e5f6g7h8-i9j0-k1l2-m3n4-o5p6q7r8s9t0",
            description="Retrieves all field-value pairs stored within a specified hash in Redis.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=RedisHashGetAllBlock.Input,
            output_schema=RedisHashGetAllBlock.Output,
            test_credentials=TEST_REDIS_CREDENTIALS,
            test_input={
                "credentials": TEST_REDIS_CREDENTIALS_INPUT,
                "host": "localhost",
                "port": 6379,
                "ssl": False,
                "key": "my_hash"
            },
            test_output=[
                ("success", True),
                ("values", {"field1": "value1", "field2": "value2", "field3": "value3"})
            ],
            test_mock={
                "run": lambda *args, **kwargs: [
                    ("success", True),
                    ("values", {"field1": "value1", "field2": "value2", "field3": "value3"})
                ]
            },
        )

    def run(
        self, input_data: Input, *, credentials: RedisCredentials, **kwargs
    ) -> BlockOutput:
        try:
            with redis.Redis(
                host=input_data.host,
                port=input_data.port,
                username=credentials.username.get_secret_value() if credentials.username else "default",
                password=credentials.password.get_secret_value() if credentials.password else None,
                ssl=input_data.ssl,
                decode_responses=True
            ) as r:
                # Get all field-value pairs
                values = r.hgetall(input_data.key)

                yield "success", True
                yield "values", values

        except Exception as e:
            yield "success", False
            yield "values", {}
            yield "error", str(e)

class RedisHashDeleteBlock(Block):
    """Removes one or more specified fields and their values from a hash."""
    class Input(BlockSchema):
        credentials: RedisCredentialsInput = RedisCredentialsField()
        host: str = SchemaField(description="Redis server host address")
        port: int = SchemaField(description="Redis server port", default=6379, advanced=False)
        ssl: bool = SchemaField(description="Whether to use SSL for the connection", default=False)
        key: str = SchemaField(description="The key of the hash.")
        fields: list[str] = SchemaField(description="Field names to delete from the hash.")

    class Output(BlockSchema):
        success: bool = SchemaField(description="True if the operation was successful.")
        deleted_count: int = SchemaField(description="Number of fields that were actually deleted.")
        error: str = SchemaField(description="Error message if operation failed.")

    def __init__(self):
        super().__init__(
            id="f6g7h8i9-j0k1-l2m3-n4o5-p6q7r8s9t0u1",
            description="Removes one or more specified fields and their values from a hash in Redis.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=RedisHashDeleteBlock.Input,
            output_schema=RedisHashDeleteBlock.Output,
            test_credentials=TEST_REDIS_CREDENTIALS,
            test_input={
                "credentials": TEST_REDIS_CREDENTIALS_INPUT,
                "host": "localhost",
                "port": 6379,
                "ssl": False,
                "key": "my_hash",
                "fields": ["field1", "field2"]
            },
            test_output=[
                ("success", True),
                ("deleted_count", 2)
            ],
            test_mock={
                "run": lambda *args, **kwargs: [
                    ("success", True),
                    ("deleted_count", 2)
                ]
            },
        )

    def run(
        self, input_data: Input, *, credentials: RedisCredentials, **kwargs
    ) -> BlockOutput:
        try:
            with redis.Redis(
                host=input_data.host,
                port=input_data.port,
                username=credentials.username.get_secret_value() if credentials.username else "default",
                password=credentials.password.get_secret_value() if credentials.password else None,
                ssl=input_data.ssl,
                decode_responses=True
            ) as r:
                # Delete the specified fields from the hash
                deleted_count = r.hdel(input_data.key, *input_data.fields)

                yield "success", True
                yield "deleted_count", deleted_count

        except Exception as e:
            yield "success", False
            yield "deleted_count", 0
            yield "error", str(e)

class RedisSetManageBlock(Block):
    """Adds or removes specified members from a set (collection of unique items)."""
    class Input(BlockSchema):
        credentials: RedisCredentialsInput = RedisCredentialsField()
        host: str = SchemaField(description="Redis server host address")
        port: int = SchemaField(description="Redis server port", default=6379, advanced=False)
        ssl: bool = SchemaField(description="Whether to use SSL for the connection", default=False)
        key: str = SchemaField(description="The key of the set to manage.")
        members: list[str] = SchemaField(description="Members to add to or remove from the set.")
        action: SetAction = SchemaField(
            description="Action to perform: ADD or REMOVE members.",
            default=SetAction.ADD
        )

    class Output(BlockSchema):
        success: bool = SchemaField(description="True if the operation was successful.")
        modified_count: int = SchemaField(description="Number of members that were added or removed.")
        error: str = SchemaField(description="Error message if operation failed.")

    def __init__(self):
        super().__init__(
            id="g7h8i9j0-k1l2-m3n4-o5p6-q7r8s9t0u1v2",
            description="Adds or removes specified members from a set in Redis.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=RedisSetManageBlock.Input,
            output_schema=RedisSetManageBlock.Output,
            test_credentials=TEST_REDIS_CREDENTIALS,
            test_input={
                "credentials": TEST_REDIS_CREDENTIALS_INPUT,
                "host": "localhost",
                "port": 6379,
                "ssl": False,
                "key": "my_set",
                "members": ["member1", "member2"],
                "action": SetAction.ADD
            },
            test_output=[
                ("success", True),
                ("modified_count", 2)
            ],
            test_mock={
                "run": lambda *args, **kwargs: [
                    ("success", True),
                    ("modified_count", 2)
                ]
            },
        )

    def run(
        self, input_data: Input, *, credentials: RedisCredentials, **kwargs
    ) -> BlockOutput:
        try:
            with redis.Redis(
                host=input_data.host,
                port=input_data.port,
                username=credentials.username.get_secret_value() if credentials.username else "default",
                password=credentials.password.get_secret_value() if credentials.password else None,
                ssl=input_data.ssl,
                decode_responses=True
            ) as r:
                modified_count = 0

                if input_data.action == SetAction.ADD:
                    # Add members to the set
                    modified_count = r.sadd(input_data.key, *input_data.members)
                else:  # SetAction.REMOVE
                    # Remove members from the set
                    modified_count = r.srem(input_data.key, *input_data.members)

                yield "success", True
                yield "modified_count", modified_count

        except Exception as e:
            yield "success", False
            yield "modified_count", 0
            yield "error", str(e)

# Having some issue
class RedisSetQueryBlock(Block):
    """Queries set members or checks membership in a set."""
    class Input(BlockSchema):
        credentials: RedisCredentialsInput = RedisCredentialsField()
        host: str = SchemaField(description="Redis server host address")
        port: int = SchemaField(description="Redis server port", default=6379, advanced=False)
        ssl: bool = SchemaField(description="Whether to use SSL for the connection", default=False)
        key: str = SchemaField(description="The key of the set to query.")
        action: SetQueryAction = SchemaField(
            description="Query action: GET_ALL to retrieve all members, IS_MEMBER to check membership.",
            default=SetQueryAction.GET_ALL
        )
        member: Optional[str] = SchemaField(
            description="Member to check (required for IS_MEMBER action).",
            default=None
        )

    class Output(BlockSchema):
        success: bool = SchemaField(description="True if the operation was successful.")
        members: list[str] = SchemaField(description="List of all members in the set (for GET_ALL action).")
        is_member: bool = SchemaField(description="True if the specified member exists in the set (for IS_MEMBER action).")
        error: str = SchemaField(description="Error message if operation failed.")

    def __init__(self):
        super().__init__(
            id="h8i9j0k1-l2m3-n4o5-p6q7-r8s9t0u1v2w3",
            description="Queries set members or checks membership in a set in Redis.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=RedisSetQueryBlock.Input,
            output_schema=RedisSetQueryBlock.Output,
            test_credentials=TEST_REDIS_CREDENTIALS,
            test_input={
                "credentials": TEST_REDIS_CREDENTIALS_INPUT,
                "host": "localhost",
                "port": 6379,
                "ssl": False,
                "key": "my_set",
                "action": SetQueryAction.GET_ALL,
                "member": None
            },
            test_output=[
                ("success", True),
                ("members", ["member1", "member2", "member3"])
            ],
            test_mock={
                "run": lambda *args, **kwargs: [
                    ("success", True),
                    ("members", ["member1", "member2", "member3"])
                ]
            },
        )

    def run(
        self, input_data: Input, *, credentials: RedisCredentials, **kwargs
    ) -> BlockOutput:
        try:
            with redis.Redis(
                host=input_data.host,
                port=input_data.port,
                username=credentials.username.get_secret_value() if credentials.username else "default",
                password=credentials.password.get_secret_value() if credentials.password else None,
                ssl=input_data.ssl,
                decode_responses=True
            ) as r:
                if input_data.action == SetQueryAction.GET_ALL:
                    members = r.smembers(input_data.key)
                    yield "success", True
                    yield "members", members
                    yield "is_member", False
                else:
                    if not input_data.member:
                        raise ValueError("Member must be specified for IS_MEMBER action")
                    # Check if member exists in the set
                    is_member = r.sismember(input_data.key, input_data.member)
                    yield "success", True
                    yield "members", []  # Default value for IS_MEMBER action
                    yield "is_member", bool(is_member)

        except Exception as e:
            yield "success", False
            yield "members", []
            yield "is_member", False
            yield "error", str(e)

class RedisPublishBlock(Block):
    """Sends (publishes) a message to a specific communication channel."""
    class Input(BlockSchema):
        credentials: RedisCredentialsInput = RedisCredentialsField()
        host: str = SchemaField(description="Redis server host address")
        port: int = SchemaField(description="Redis server port", default=6379, advanced=False)
        ssl: bool = SchemaField(description="Whether to use SSL for the connection", default=False)
        channel: str = SchemaField(description="The channel to publish the message to.")
        message: str = SchemaField(description="The message to publish.")

    class Output(BlockSchema):
        success: bool = SchemaField(description="True if the operation was successful.")
        receivers: int = SchemaField(description="Number of clients that received the message.")
        error: str = SchemaField(description="Error message if operation failed.")

    def __init__(self):
        super().__init__(
            id="i9j0k1l2-m3n4-o5p6-q7r8-s9t0u1v2w3x4",
            description="Sends (publishes) a message to a specific communication channel in Redis.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=RedisPublishBlock.Input,
            output_schema=RedisPublishBlock.Output,
            test_credentials=TEST_REDIS_CREDENTIALS,
            test_input={
                "credentials": TEST_REDIS_CREDENTIALS_INPUT,
                "host": "localhost",
                "port": 6379,
                "ssl": False,
                "channel": "my_channel",
                "message": "Hello Redis!"
            },
            test_output=[
                ("success", True),
                ("receivers", 1)
            ],
            test_mock={
                "run": lambda *args, **kwargs: [
                    ("success", True),
                    ("receivers", 1)
                ]
            },
        )

    def run(
        self, input_data: Input, *, credentials: RedisCredentials, **kwargs
    ) -> BlockOutput:
        try:
            with redis.Redis(
                host=input_data.host,
                port=input_data.port,
                username=credentials.username.get_secret_value() if credentials.username else "default",
                password=credentials.password.get_secret_value() if credentials.password else None,
                ssl=input_data.ssl,
                decode_responses=True
            ) as r:
                # Publish message to the specified channel
                receivers = r.publish(input_data.channel, input_data.message)

                yield "success", True
                yield "receivers", receivers

        except Exception as e:
            yield "success", False
            yield "receivers", 0
            yield "error", str(e)
