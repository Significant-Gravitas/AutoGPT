## Class WeaviateMemory

This class is a memory provider singleton that uses Weaviate, an open-source search engine that allows for efficient semantic search based on vector embeddings.

### Method `__init__(self, cfg)`

#### Arguments
- `cfg`(`Config`): An instance of `Config` class.

#### Description
- This method initializes the WeaviateMemory class and creates a connection to Weaviate based on the provided configuration.

### Method `format_classname(index)`

#### Arguments
- `index`(`str`): A string that represents the index name.

#### Description
- This static method takes the `index` string argument and formats it according to Weaviate convention as per the provided naming rules.

### Method `_create_schema(self)`

#### Description
- If the schema of the `self.index` does not already exist, this method creates one when called.

### Method `_build_auth_credentials(self, cfg)`

#### Arguments
- `cfg`(`Config`): An instance of `Config` class.

#### Description
- This method builds the authorization credentials required to connect to Weaviate using the configuration params provided.

### Method `add(self, data)`

#### Arguments
- `data`(`str`): a string containing the data which needs to be added.

#### Description
- This method adds the `data` to the Weaviate database after generating a vector representation.

### Method `get(self, data)`

#### Arguments
- `data`(`str`): a string containing the data which needs to be fetched.

#### Description
- This method returns the relevant data for the input `data` after generating a vector representation.

### Method `clear(self)`

#### Description
- This method deletes all the data stored in the weaviate index.

### Method `get_relevant(self, data, num_relevant=5)`

#### Arguments
- `data`(`str`): a string containing the data in relation to which relevant data needs to be fetched.
- `num_relevant`(`int`): Number of relevant data points to return. Default is 5.

#### Description
- This method returns the `num_relevant` relevant data for the input `data` after generating a vector representation.

### Method `get_stats(self)`

#### Description
- This method returns the metadata about the stored data in the Weaviate index.

### Example
```python
cfg = Config()
mem = WeaviateMemory(cfg)
data = 'Artificial Intelligence'
print(mem.add(data))
print(mem.get(data))
print(mem.get_relevant(data))
print(mem.clear())
```