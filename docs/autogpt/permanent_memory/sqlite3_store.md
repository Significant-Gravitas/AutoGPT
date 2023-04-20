## MemoryDB Class

The `MemoryDB` class is a wrapper around a SQLite database that can be used for storing blocks of text data. The class initializes the database and provides methods for inserting, updating, and searching text data.

### Class Methods

#### `__init__(self, db=None)`

The constructor for the `MemoryDB` class takes an optional argument for the name of the database file. If a file name is not provided, the database will default to a file named "mem.sqlite3" in the current working directory. If a connection to the database cannot be established, then the database will be opened in dynamic memory, and not be persistent. The method creates a `FTS5` virtual table with columns `session`, `key`, and `block` which will be used to store text.

#### `get_cnx(self)`

This method returns the connection object to the database.

#### `get_max_session_id(self)`

This method returns the highest session id from the database. 

#### `get_next_key(self)`

This method returns the next unused `key` value for inserting a new text block into the database.

#### `insert(self, text=None)`

This method inserts a new block of text data into the database with a unique `key` value. If `text` is `None`, nothing will be inserted.

#### `overwrite(self, key, text)`

This method overwrites previously inserted text at the given `key` in the database with the new `text` value.

#### `delete_memory(self, key, session_id=None)`

This method deletes the block of text data identified by the `key`. If `session_id` is not provided, it will use the current session id.

#### `search(self, text)`

This method searches the entire database for the given `text` argument and returns a list of lines containing the searched text.

#### `get_session(self, id=None)`

This method returns all the blocks of text added during the current session if `id=None`. If a session id is provided, it returns all the text blocks in that session.

#### `quit(self)`

This method commits any pending changes and closes the database connection.

### Global Variable

This module contains one global variable, `permanent_memory`, which is an instance of `MemoryDB`. This object is created with default values when the module is imported and can be used to store data persistently across multiple runs of a program. 

## Example

```python
from memory_db import MemoryDB

# create a temporary memory database
temp_memory = MemoryDB()

# insert some data
temp_memory.insert('the quick brown fox')
temp_memory.insert('jumped over')
temp_memory.insert('the lazy dog')

# search for data
result = temp_memory.search('jumped')
assert result[0] == 'jumped over'

# delete a record
temp_memory.delete_memory(1)
result = temp_memory.get_session()
assert result == ['the quick brown fox', 'the lazy dog']

# update a record
temp_memory.overwrite(0, 'a quick brown fox')
result = temp_memory.get_session()
assert result == ['a quick brown fox', 'the lazy dog']

# quit
temp_memory.quit()
```