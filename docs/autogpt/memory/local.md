## `LocalCache` class

A class that stores the memory in a local file.

### Methods

#### `__init__(self, cfg) -> None`

Initialize a class instance.

##### Input
- `cfg`: Config object

##### Output
- `None`

##### Example

```python
cfg = Config()
lc = LocalCache(cfg)
```

---
#### `add(self, text: str)`

Add text to our list of texts, add embedding as row to our embeddings-matrix.

##### Input
- `text`: str

Output
- `None`

##### Example

```python
lc.add("How do I train a deep neural net?")
```

---
#### `clear(self) -> str`

Clears the redis server.

##### Output
- A message indicating that the memory has been cleared.

##### Example

```python
lc.clear()
```

---
#### `get(self, data: str) -> list | None`

Gets the data from the memory that is most relevant to the given data.

##### Input
- `data`: The data to compare to.

##### Output
- The most relevant data.

##### Example

```python
lc.get("How do I train a deep neural net?")
```

---
#### `get_relevant(self, text: str, k: int) -> list`

Matrix-vector mult to find score-for-each-row-of-matrix, get indices for top-k winning scores, return texts for those indices.

##### Input
- `text`: str
- `k`: int

##### Output
- List of texts

##### Example

```python
lc.get_relevant("How do I train a deep neural net?", 3)
```

---
#### `get_stats(self) -> tuple`

Returns the stats of the local cache.

##### Output
- A tuple containing:
     - Number of texts stored
     - An tuple of integers representing the shape of stored embeddings.

##### Example

```python
lc.get_stats()
```