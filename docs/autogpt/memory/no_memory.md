# NoMemory class

This class is a part of the `autogpt` package and is used as a default memory provider. It does not store any data and serves as a placeholder to be easily substituted with other memory providers that store data.

## Methods

### `__init__(self, cfg)`

Initializes the NoMemory provider.

**Args**:
- `cfg`: The config object.

### `add(self, data: str) -> str`

Adds a data point to the memory. No action is taken in NoMemory.

**Args**:
- `data` (str): The data to add.

**Returns**: An empty string.

### `get(self, data: str) -> list[Any] | None`

Gets the data from the memory that is most relevant to the given data. NoMemory always returns None.

**Args**:
- `data` (str): The data to compare to.

**Returns**: None.

### `clear(self) -> str`

Clears the memory. No action is taken in NoMemory.

**Returns**: An empty string.

### `get_relevant(self, data: str, num_relevant: int = 5) -> list[Any] | None`

Returns all the data in the memory that is relevant to the given data. NoMemory always returns None.

**Args**:
- `data` (str): The data to compare to.
- `num_relevant` (int): The number of relevant data to return.

**Returns**: None.

### `get_stats(self)`

Returns an empty dictionary as there are no stats in NoMemory.

**Returns**: An empty dictionary.