# Contributing to AutoGPT Agent Server: Creating and Testing Blocks

This guide will walk you through the process of creating and testing a new block for the AutoGPT Agent Server, using the WikipediaSummaryBlock as an example.

## Understanding Blocks and Testing

Blocks are reusable components that can be connected to form a graph representing an agent's behavior. Each block has inputs, outputs, and a specific function. Proper testing is crucial to ensure blocks work correctly and consistently.

## Creating and Testing a New Block

Follow these steps to create and test a new block:

1. **Create a new Python file** in the `autogpt_server/blocks` directory. Name it descriptively and use snake_case. For example: `get_wikipedia_summary.py`.

2. **Import necessary modules and create a class that inherits from `Block`**. Make sure to include all necessary imports for your block.

 Every block should contain the following:

   ```python
   from autogpt_server.data.block import Block, BlockSchema, BlockOutput
   ```

   Example for the Wikipedia summary block:

   ```python
   from autogpt_server.data.block import Block, BlockSchema, BlockOutput
   from autogpt_server.utils.get_request import GetRequest
   import requests

   class WikipediaSummaryBlock(Block, GetRequest):
       # Block implementation will go here
   ```

3. **Define the input and output schemas** using `BlockSchema`. These schemas specify the data structure that the block expects to receive (input) and produce (output).

   - The input schema defines the structure of the data the block will process. Each field in the schema represents a required piece of input data.
   - The output schema defines the structure of the data the block will return after processing. Each field in the schema represents a piece of output data.

   Example:

   ```python
   class Input(BlockSchema):
       topic: str  # The topic to get the Wikipedia summary for

   class Output(BlockSchema):
       summary: str  # The summary of the topic from Wikipedia
       error: str  # Any error message if the request fails
   ```

4. **Implement the `__init__` method, including test data and mocks:**

   ```python
   def __init__(self):
       super().__init__(
           # Unique ID for the block
           # you can generate this with this python one liner
           # print(__import__('uuid').uuid4())
           id="h5e7f8g9-1b2c-3d4e-5f6g-7h8i9j0k1l2m",
           input_schema=WikipediaSummaryBlock.Input,  # Assign input schema
           output_schema=WikipediaSummaryBlock.Output,  # Assign output schema

            # Provide sample input, output and test mock for testing the block

           test_input={"topic": "Artificial Intelligence"},
           test_output=("summary", "summary content"),
           test_mock={"get_request": lambda url, json: {"extract": "summary content"}},
       )
   ```

   - `id`: A unique identifier for the block.

   - `input_schema` and `output_schema`: Define the structure of the input and output data.

   Let's break down the testing components:

   - `test_input`: This is a sample input that will be used to test the block. It should be a valid input according to your Input schema.

   - `test_output`: This is the expected output when running the block with the `test_input`. It should match your Output schema. For non-deterministic outputs or when you only want to assert the type, you can use Python types instead of specific values. In this example, `("summary", str)` asserts that the output key is "summary" and its value is a string.

   - `test_mock`: This is crucial for blocks that make network calls. It provides a mock function that replaces the actual network call during testing. 

     In this case, we're mocking the `get_request` method to always return a dictionary with an 'extract' key, simulating a successful API response. This allows us to test the block's logic without making actual network requests, which could be slow, unreliable, or rate-limited.

5. **Implement the `run` method with error handling:**, this should contain the main logic of the block:

   ```python
   def run(self, input_data: Input) -> BlockOutput:
       try:
           topic = input_data.topic
           url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"

           response = self.get_request(url, json=True)
           yield "summary", response['extract']

       except requests.exceptions.HTTPError as http_err:
           yield "error", f"HTTP error occurred: {http_err}"
       except requests.RequestException as e:
           yield "error", f"Request to Wikipedia failed: {e}"
       except KeyError as e:
           yield "error", f"Error parsing Wikipedia response: {e}"
   ```

   - **Try block**: Contains the main logic to fetch and process the Wikipedia summary.
   - **API request**: Send a GET request to the Wikipedia API.
   - **Error handling**: Handle various exceptions that might occur during the API request and data processing.
   - **Yield**: Use `yield` to output the results.

## Key Points to Remember

- **Unique ID**: Give your block a unique ID in the **init** method.
- **Input and Output Schemas**: Define clear input and output schemas.
- **Error Handling**: Implement error handling in the `run` method.
- **Output Results**: Use `yield` to output results in the `run` method.
- **Testing**: Provide test input and output in the **init** method for automatic testing.

## Understanding the Testing Process

The testing of blocks is handled by `test_block.py`, which does the following:

1. It calls the block with the provided `test_input`.
2. If a `test_mock` is provided, it temporarily replaces the specified methods with the mock functions.
3. It then asserts that the output matches the `test_output`.

For the WikipediaSummaryBlock:

- The test will call the block with the topic "Artificial Intelligence".
- Instead of making a real API call, it will use the mock function, which returns `{"extract": "summary content"}`.
- It will then check if the output key is "summary" and its value is a string.

This approach allows us to test the block's logic comprehensively without relying on external services, while also accommodating non-deterministic outputs.

## Tips for Effective Block Testing

1. **Provide realistic test_input**: Ensure your test input covers typical use cases.

2. **Define appropriate test_output**: 
   - For deterministic outputs, use specific expected values.
   - For non-deterministic outputs or when only the type matters, use Python types (e.g., `str`, `int`, `dict`).
   - You can mix specific values and types, e.g., `("key1", str), ("key2", 42)`.

3. **Use test_mock for network calls**: This prevents tests from failing due to network issues or API changes.

4. **Consider omitting test_mock for blocks without external dependencies**: If your block doesn't make network calls or use external resources, you might not need a mock.

5. **Consider edge cases**: Include tests for potential error conditions in your `run` method.

6. **Update tests when changing block behavior**: If you modify your block, ensure the tests are updated accordingly.

By following these steps, you can create new blocks that extend the functionality of the AutoGPT Agent Server.
