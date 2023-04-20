# Redis Memory Provider

This module provides a Redis memory provider which can be used for storing and retrieving data from Redis. The module contains a `RedisMemory` class with methods for adding and getting data from the memory. The `RedisMemory` class is a singleton class that is used throughout the code.

## Usage

To use the Redis memory provider:

1. Create an instance of the `RedisMemory` class with a `cfg` object as a parameter.
2. Call the `add` method with a string to add the data to the memory.
3. Call the `get` or `get_relevant` method with a string to retrieve data from the memory.

## Example

```python
from autogpt.memory.redis_memory import RedisMemory
from autogpt.cfg import RedisConfig

redis_config = RedisConfig(
    redis_host="localhost",
    redis_port=6379,   
    redis_password="password",
    wipe_redis_on_start=True,
    memory_index="my_index"
)

memory = RedisMemory(redis_config)

data = "example data"
memory.add(data)

relevant_data = memory.get_relevant(data)
print(relevant_data)
```

## Methods

### `__init__(self, cfg)`

Initializes the Redis memory provider.

Arguments:
- `cfg`: a `RedisConfig` object.

Returns:
- None.

### `add(self, data)`

Adds a data point to the memory.

Arguments:
- `data` (str): The data to add.

Returns:
- A message indicating that the data has been added.

### `get(self, data)`

Gets the data from the memory that is most relevant to the given data.

Arguments:
- `data` (str): The data to compare to.

Returns:
- The most relevant data.

### `clear(self)`

Clears the redis server.

Returns:
- A message indicating that the memory has been cleared.

### `get_relevant(self, data, num_relevant=5)`

Returns all the data in the memory that is relevant to the given data.

Arguments:
- `data` (str): The data to compare to.
- `num_relevant` (int): The number of relevant data to return.

Returns:
- A list of the most relevant data.

### `get_stats(self)`

Returns the stats of the memory index.

Returns:
- The stats of the memory index.