# DB Class
The DB class represents a simple database that stores its data as files in a directory. It is a key-value store, where keys are filenames and values are file contents. The DB class is defined in the `gpt_engineer/db.py` file.

<br>

### DB Class
Methods and how they are being used:

`__init__(self, path)`: The constructor takes a path as an argument and creates a directory at that path if it does not already exist.

`__contains__(self, key)`: This method checks if a key (filename) exists in the database. It returns True if the file exists and False otherwise.

`__getitem__(self, key)`: This method gets the value (file content) associated with a key (filename). It raises a `KeyError` if the key does not exist in the database.

`__setitem__(self, key, val)`: This method sets the value (file content) associated with a key (filename). It creates the file if it does not already exist. The value must be either a string or bytes.

<br>

### DBs Class
The DBs class is a dataclass that contains instances of the DB class for different types of data:

Each instance of the DBs class contains five databases currently:

`memory`: This database is used to store the AI's memory.
`logs`: This database is used to store logs of the AI's actions.
`preprompts`: This database is used to store preprompts that guide the AI's actions.
`input`: This database is used to store the user's input.
`workspace`: This database is used to store the AI's workspace, which includes the code it generates.

<br>

## Conclusion
The DB and DBs classes provide a simple and flexible way to manage data in the GPT-Engineer system. They allow the system to store and retrieve data as files in a directory, which makes it easy to inspect and modify the data.
