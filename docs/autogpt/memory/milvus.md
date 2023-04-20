# Milvus memory storage provider

This module contains the `MilvusMemory` class that implements an embedding memory storage provider using a Milvus server.

## Methods

### \_\_init\_\_(self, cfg)

Class constructor that connects to the Milvus server, creates a collection and index.

**Arguments**
- `cfg`: An object of `Config` class, which is the auto-gpt global config.

### add(self, data)

Adds an embedding of data into memory.

**Arguments**
- `data`: The raw text to construct embedding index.

**Returns**
- A string message.

### get(self, data)

Returns the most relevant data in memory.

**Arguments**
- `data`: The data to compare to.

### clear(self)

Drops the index in memory.

**Returns**
- A string message.

### get_relevant(self, data, num_relevant=5)

Returns the top `num_relevant` relevant data in memory.

**Arguments**
- `data`: The data to compare to.
- `num_relevant`: (Optional) The max number of relevant data. Defaults to 5.

**Returns**
- A list of the top `num_relevant` relevant data.

### get_stats(self)

Returns the stats of the Milvus cache.

**Returns**
- A string message.