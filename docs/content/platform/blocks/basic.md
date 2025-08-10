# Basic Operations Blocks

## Store Value

### What it is
A basic block that stores and forwards a value.

### What it does
This block takes an input value and stores it, allowing it to be reused without changes.

### How it works
It accepts an input value and optionally a data value. If a data value is provided, it is used as the output. Otherwise, the input value is used as the output.

### Inputs
| Input | Description |
|-------|-------------|
| Input | The value to be stored or forwarded |
| Data | An optional constant value to be stored instead of the input |

### Outputs
| Output | Description |
|--------|-------------|
| Output | The stored value (either the input or the data) |

### Possible use case
Storing a user's name at the beginning of a workflow to use it in multiple subsequent blocks without asking for it again.

---

## Print to Console

### What it is
A basic block that prints text to the console for debugging purposes.

### What it does
This block takes a text input and prints it to the console, then outputs a status message.

### How it works
It receives a text input, prints it to the console with a "Print: " prefix, and then yields a "printed" status.

### Inputs
| Input | Description |
|-------|-------------|
| Text | The text to be printed to the console |

### Outputs
| Output | Description |
|--------|-------------|
| Status | A message indicating that the text was printed ("printed") |

### Possible use case
Debugging a workflow by printing intermediate results or messages at various stages.

---

## Find in Dictionary

### What it is
A basic block that looks up a value in a dictionary, object, or list using a given key.

### What it does
This block searches for a specified key in the input data structure and returns the corresponding value if found.

### How it works
It accepts an input (dictionary, object, or list) and a key. It then attempts to find the key in the input and return the corresponding value. If the key is not found, it returns the entire input as "missing".

### Inputs
| Input | Description |
|-------|-------------|
| Input | The dictionary, object, or list to search in |
| Key | The key to look up in the input |

### Outputs
| Output | Description |
|--------|-------------|
| Output | The value found for the given key |
| Missing | The entire input if the key was not found |

### Possible use case
Extracting specific information from a complex data structure, such as finding a user's email address in a user profile dictionary.

---

## Agent Input

### What it is
An input block that provides a way to accept user input in a workflow.

### What it does
This block allows users to input values into the workflow, with options for naming, describing, and setting placeholder values.

### How it works
It accepts a value from the user, along with metadata such as name, description, and optional placeholder values. The block then outputs the provided value.

### Inputs
| Input | Description |
|-------|-------------|
| Value | The actual input value provided by the user |
| Name | A name for the input field |
| Description | An optional description of the input |
| Placeholder Values | Optional list of suggested values |
| Limit to Placeholder Values | Option to restrict input to placeholder values only |

### Outputs
| Output | Description |
|--------|-------------|
| Result | The value provided as input |

### Possible use case
Collecting user preferences at the start of a personalized recommendation workflow.

---

## Agent Output

### What it is
An output block that records and formats the final results of a workflow.

### What it does
This block takes a value and associated metadata, optionally formats it, and presents it as the output of the workflow.

### How it works
It accepts an input value along with a name, description, and optional format string. If a format string is provided, it attempts to apply the formatting to the input value before outputting it.

### Inputs
| Input | Description |
|-------|-------------|
| Value | The value to be recorded as output |
| Name | A name for the output |
| Description | An optional description of the output |
| Format | An optional format string to apply to the value |

### Outputs
| Output | Description |
|--------|-------------|
| Output | The formatted (if applicable) output value |

### Possible use case
Presenting the final results of a data analysis workflow in a specific format.

---

## Add to Dictionary

### What it is
A basic block that adds a new key-value pair to a dictionary.

### What it does
This block takes an existing dictionary (or creates a new one), a key, and a value, and adds the key-value pair to the dictionary.

### How it works
It accepts an optional input dictionary, a key, and a value. If no dictionary is provided, it creates a new one. It then adds the key-value pair to the dictionary and returns the updated dictionary.

### Inputs
| Input | Description |
|-------|-------------|
| Dictionary | An optional existing dictionary to add to |
| Key | The key for the new entry |
| Value | The value for the new entry |

### Outputs
| Output | Description |
|--------|-------------|
| Updated Dictionary | The dictionary with the new entry added |
| Error | An error message if the operation fails |

### Possible use case
Building a user profile by gradually adding new information as it's collected throughout a workflow.

---

## Add to List

### What it is
A basic block that adds a new entry to a list.

### What it does
This block takes an existing list (or creates a new one) and adds a new entry to it, optionally at a specified position.

### How it works
It accepts an optional input list, an entry to add, and an optional position. If no list is provided, it creates a new one. It then adds the entry to the list at the specified position (or at the end if no position is given) and returns the updated list.

### Inputs
| Input | Description |
|-------|-------------|
| List | An optional existing list to add to |
| Entry | The new item to add to the list |
| Position | An optional position to insert the new entry |

### Outputs
| Output | Description |
|--------|-------------|
| Updated List | The list with the new entry added |
| Error | An error message if the operation fails |

### Possible use case
Maintaining a to-do list in a task management workflow, where new tasks can be added at specific priorities (positions).

---

## Note

### What it is
A basic block that displays a sticky note with custom text.

### What it does
This block takes a text input and displays it as a sticky note in the workflow interface.

### How it works
It simply accepts a text input and passes it through as an output to be displayed as a note.

### Inputs
| Input | Description |
|-------|-------------|
| Text | The text to display in the sticky note |

### Outputs
| Output | Description |
|--------|-------------|
| Output | The text to display in the sticky note |

### Possible use case
Adding explanatory notes or reminders within a complex workflow to help users understand different stages or provide additional context.