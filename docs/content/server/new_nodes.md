# Contributing to AutoGPT Agent Server: Creating Nodes!

This guide will walk you through the process of creating a new node (also called a block) for the AutoGPT Agent Server. We'll use the GetWikipediaSummary node as an example.

## Understanding Nodes

Nodes are reusable components that can be connected to form a graph representing an agent's behavior. Each node has inputs, outputs, and a specific function it performs.

## Creating a New Node

To create a new node, follow these steps:

1. **Create a new Python file** in the `autogpt_server/blocks` directory. Name it descriptively and use snake_case. For example: `get_wikipedia_summary.py`.

2. **Import necessary modules and create a class that inherits from `Block`**. Make sure to include all necessary imports for your node. Every node should contain:

   ```python
   from autogpt_server.data.block import Block, BlockSchema, BlockOutput
   ```

   Example for the Wikipedia summary node:

   ```python
   import requests
   from autogpt_server.data.block import Block, BlockSchema, BlockOutput

   class GetWikipediaSummary(Block):
       # Node implementation will go here
   ```

3. **Define the input and output schemas** using `BlockSchema`. These schemas specify the data structure that the node expects to receive (input) and produce (output).

   - The input schema defines the structure of the data the node will process. Each field in the schema represents a required piece of input data.
   - The output schema defines the structure of the data the node will return after processing. Each field in the schema represents a piece of output data.

   Example:

   ```python
   class Input(BlockSchema):
       topic: str  # The topic to get the Wikipedia summary for

   class Output(BlockSchema):
       summary: str  # The summary of the topic from Wikipedia
   ```

4. **Implement the `__init__` method**:

   ```python
   def __init__(self):
       super().__init__(
           id="h5e7f8g9-1b2c-3d4e-5f6g-7h8i9j0k1l2m",  # Unique ID for the node
           input_schema=GetWikipediaSummary.Input,  # Assign input schema
           output_schema=GetWikipediaSummary.Output,  # Assign output schema

           # Provide sample input and output for testing the node

           test_input={"topic": "Artificial Intelligence"},
           test_output={"summary": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals."},
       )
   ```

   - `id`: A unique identifier for the node.
   - `input_schema` and `output_schema`: Define the structure of the input and output data.
   - `test_input` and `test_output`: Provide sample input and output data for testing the node.

5. **Implement the `run` method**, which contains the main logic of the node:

   ```python
   def run(self, input_data: Input) -> BlockOutput:
       try:
           # Make the request to Wikipedia API
           response = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{input_data.topic}")
           response.raise_for_status()
           summary_data = response.json()

           # Output the summary
           yield "summary", summary_data['extract']

       except requests.exceptions.HTTPError as http_err:
           raise ValueError(f"HTTP error occurred: {http_err}")
       except requests.RequestException as e:
           raise ValueError(f"Request to Wikipedia API failed: {e}")
       except KeyError as e:
           raise ValueError(f"Error processing Wikipedia data: {e}")
   ```

   - **Try block**: Contains the main logic to fetch and process the Wikipedia summary.
   - **API request**: Send a GET request to the Wikipedia API.
   - **Error handling**: Handle various exceptions that might occur during the API request and data processing.
   - **Yield**: Use `yield` to output the results.

6. **Register the new node** by adding it to the `__init__.py` file in the `autogpt_server/blocks` directory. This step makes your new node available to the rest of the server.

   - Open the `__init__.py` file in the `autogpt_server/blocks` directory.
   - Add an import statement for your new node at the top of the file.
   - Add the new node to the `AVAILABLE_BLOCKS` and `__all__` lists.

   Example:

   ```python
   from autogpt_server.blocks import sample, reddit, text, ai, wikipedia, discord, get_wikipedia_summary  # Import your new node
   from autogpt_server.data.block import Block

   AVAILABLE_BLOCKS = {
       block.id: block
       for block in [v() for v in Block.__subclasses__()]
   }

   __all__ = ["ai", "sample", "reddit", "text", "AVAILABLE_BLOCKS", "wikipedia", "discord", "get_wikipedia_summary"]
   ```

   - The import statement ensures your new node is included in the module.
   - The `AVAILABLE_BLOCKS` dictionary includes all blocks by their ID.
   - The `__all__` list specifies all public objects that the module exports.

### Full Code example

Here is the complete implementation of the `GetWikipediaSummary` nodes:

```python
import requests
from autogpt_server.data.block import Block, BlockSchema, BlockOutput

class GetWikipediaSummary(Block):
    # Define the input schema with the required field 'topic'
    class Input(BlockSchema):
        topic: str  # The topic to get the Wikipedia summary for

    # Define the output schema with the field 'summary'
    class Output(BlockSchema):
        summary: str  # The summary of the topic from Wikipedia

    def __init__(self):
        super().__init__(
            id="h5e7f8g9-1b2c-3d4e-5f6g-7h8i9j0k1l2m",  # Unique ID for the node
            input_schema=GetWikipediaSummary.Input,  # Assign input schema
            output_schema=GetWikipediaSummary.Output,  # Assign output schema

            # Provide sample input and output for testing the node

            test_input={"topic": "Artificial Intelligence"},
            test_output={"summary": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals."},
        )

    def run(self, input_data: Input) -> BlockOutput:
        try:
            # Make the request to Wikipedia API
            response = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{input_data.topic}")
            response.raise_for_status()
            summary_data = response.json()

            # Output the summary
            yield "summary", summary_data['extract']

        except requests.exceptions.HTTPError as http_err:
            raise ValueError(f"HTTP error occurred: {http_err}")
        except requests.RequestException as e:
            raise ValueError(f"Request to Wikipedia API failed: {e}")
        except KeyError as e:
            raise ValueError(f"Error processing Wikipedia data: {e}")
```

## Key Points to Remember

- **Unique ID**: Give your node a unique ID in the `__init__` method.
- **Input and Output Schemas**: Define clear input and output schemas.
- **Error Handling**: Implement error handling in the `run` method.
- **Output Results**: Use `yield` to output results in the `run` method.
- **Register the Node**: Add your new node to the `__init__.py` file to make it available to the server.
- **Testing**: Provide test input and output in the `__init__` method for automatic testing.

By following these steps, you can create new nodes that extend the functionality of the AutoGPT Agent Server.
