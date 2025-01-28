# Contributing to AutoGPT Agent Server: Creating and Testing Blocks

This guide will walk you through the process of creating and testing a new block for the AutoGPT Agent Server, using the WikipediaSummaryBlock as an example.

## Understanding Blocks and Testing

Blocks are reusable components that can be connected to form a graph representing an agent's behavior. Each block has inputs, outputs, and a specific function. Proper testing is crucial to ensure blocks work correctly and consistently.

## Creating and Testing a New Block

Follow these steps to create and test a new block:

1. **Create a new Python file** for your block in the `autogpt_platform/backend/backend/blocks` directory. Name it descriptively and use snake_case. For example: `get_wikipedia_summary.py`.

2. **Import necessary modules and create a class that inherits from `Block`**. Make sure to include all necessary imports for your block.

    Every block should contain the following:

    ```python
    from backend.data.block import Block, BlockSchema, BlockOutput
    ```

    Example for the Wikipedia summary block:

    ```python
    from backend.data.block import Block, BlockSchema, BlockOutput
    from backend.utils.get_request import GetRequest
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
        error: str  # Any error message if the request fails, error field needs to be named `error`.
    ```

4. **Implement the `__init__` method, including test data and mocks:**

    !!! important
         Use UUID generator (e.g. https://www.uuidgenerator.net/) for every new block `id` and *do not* make up your own. Alternatively, you can run this python code to generate an uuid: `print(__import__('uuid').uuid4())`

    ```python
    def __init__(self):
        super().__init__(
            # Unique ID for the block, used across users for templates
            # If you are an AI leave it as is or change to "generate-proper-uuid"
            id="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
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

5. **Implement the `run` method with error handling.** This should contain the main logic of the block:

   ```python
   def run(self, input_data: Input, **kwargs) -> BlockOutput:
       try:
           topic = input_data.topic
           url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"

           response = self.get_request(url, json=True)
           yield "summary", response['extract']

       except requests.exceptions.HTTPError as http_err:
           raise RuntimeError(f"HTTP error occurred: {http_err}")
   ```

   - **Try block**: Contains the main logic to fetch and process the Wikipedia summary.
   - **API request**: Send a GET request to the Wikipedia API.
   - **Error handling**: Handle various exceptions that might occur during the API request and data processing. We don't need to catch all exceptions, only the ones we expect and can handle. The uncaught exceptions will be automatically yielded as `error` in the output. Any block that raises an exception (or yields an `error` output) will be marked as failed. Prefer raising exceptions over yielding `error`, as it will stop the execution immediately.
   - **Yield**: Use `yield` to output the results. Prefer to output one result object at a time. If you are calling a function that returns a list, you can yield each item in the list separately. You can also yield the whole list as well, but do both rather than yielding the list. For example: If you were writing a block that outputs emails, you'd yield each email as a separate result object, but you could also yield the whole list as an additional single result object. Yielding output named `error` will break the execution right away and mark the block execution as failed.
   - **kwargs**: The `kwargs` parameter is used to pass additional arguments to the block. It is not used in the example above, but it is available to the block. You can also have args as inline signatures in the run method ala `def run(self, input_data: Input, *, user_id: str, **kwargs) -> BlockOutput:`.
       Available kwargs are:
       - `user_id`: The ID of the user running the block.
       - `graph_id`: The ID of the agent that is executing the block. This is the same for every version of the agent
       - `graph_exec_id`: The ID of the execution of the agent. This changes every time the agent has a new "run"
       - `node_exec_id`: The ID of the execution of the node. This changes every time the node is executed
       - `node_id`: The ID of the node that is being executed. It changes every version of the graph, but not every time the node is executed.

### Field Types

#### oneOf fields
`oneOf` allows you to specify that a field must be exactly one of several possible options. This is useful when you want your block to accept different types of inputs that are mutually exclusive.

Example:
```python
attachment: Union[Media, DeepLink, Poll, Place, Quote] = SchemaField(
    discriminator='discriminator',
    description="Attach either media, deep link, poll, place or quote - only one can be used"
)
```

The `discriminator` parameter tells AutoGPT which field to look at in the input to determine which type it is.

In each model, you need to define the discriminator value:
```python
class Media(BaseModel):
    discriminator: Literal['media']
    media_ids: List[str]

class DeepLink(BaseModel):
    discriminator: Literal['deep_link']
    direct_message_deep_link: str
```

#### OptionalOneOf fields
`OptionalOneOf` is similar to `oneOf` but allows the field to be optional (None). This means the field can be either one of the specified types or None.

Example:
```python
attachment: Union[Media, DeepLink, Poll, Place, Quote] | None = SchemaField(
    discriminator='discriminator',
    description="Optional attachment - can be media, deep link, poll, place, quote or None"
)
```

The key difference is the `| None` which makes the entire field optional.

### Blocks with authentication

Our system supports auth offloading for API keys and OAuth2 authorization flows.
Adding a block with API key authentication is straight-forward, as is adding a block
for a service that we already have OAuth2 support for.

Implementing the block itself is relatively simple. On top of the instructions above,
you're going to add a `credentials` parameter to the `Input` model and the `run` method:

```python
from backend.data.model import (
    APIKeyCredentials,
    OAuth2Credentials,
    Credentials,
)

from backend.data.block import Block, BlockOutput, BlockSchema
from backend.data.model import CredentialsField
from backend.integrations.providers import ProviderName


# API Key auth:
class BlockWithAPIKeyAuth(Block):
    class Input(BlockSchema):
        # Note that the type hint below is require or you will get a type error.
        # The first argument is the provider name, the second is the credential type.
        credentials: CredentialsMetaInput[
            Literal[ProviderName.GITHUB], Literal["api_key"]
        ] = CredentialsField(
            description="The GitHub integration can be used with "
            "any API key with sufficient permissions for the blocks it is used on.",
        )

    # ...

    def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        ...

# OAuth:
class BlockWithOAuth(Block):
    class Input(BlockSchema):
        # Note that the type hint below is require or you will get a type error.
        # The first argument is the provider name, the second is the credential type.
        credentials: CredentialsMetaInput[
            Literal[ProviderName.GITHUB], Literal["oauth2"]
        ] = CredentialsField(
            required_scopes={"repo"},
            description="The GitHub integration can be used with OAuth.",
        )

    # ...

    def run(
        self,
        input_data: Input,
        *,
        credentials: OAuth2Credentials,
        **kwargs,
    ) -> BlockOutput:
        ...

# API Key auth + OAuth:
class BlockWithAPIKeyAndOAuth(Block):
    class Input(BlockSchema):
        # Note that the type hint below is require or you will get a type error.
        # The first argument is the provider name, the second is the credential type.
        credentials: CredentialsMetaInput[
            Literal[ProviderName.GITHUB], Literal["api_key", "oauth2"]
        ] = CredentialsField(
            required_scopes={"repo"},
            description="The GitHub integration can be used with OAuth, "
            "or any API key with sufficient permissions for the blocks it is used on.",
        )

    # ...

    def run(
        self,
        input_data: Input,
        *,
        credentials: Credentials,
        **kwargs,
    ) -> BlockOutput:
        ...
```

The credentials will be automagically injected by the executor in the back end.

The `APIKeyCredentials` and `OAuth2Credentials` models are defined [here](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform/autogpt_libs/autogpt_libs/supabase_integration_credentials_store/types.py).
To use them in e.g. an API request, you can either access the token directly:

```python
# credentials: APIKeyCredentials
response = requests.post(
    url,
    headers={
        "Authorization": f"Bearer {credentials.api_key.get_secret_value()})",
    },
)

# credentials: OAuth2Credentials
response = requests.post(
    url,
    headers={
        "Authorization": f"Bearer {credentials.access_token.get_secret_value()})",
    },
)
```

or use the shortcut `credentials.auth_header()`:

```python
# credentials: APIKeyCredentials | OAuth2Credentials
response = requests.post(
    url,
    headers={"Authorization": credentials.auth_header()},
)
```

The `ProviderName` enum is the single source of truth for which providers exist in our system.
Naturally, to add an authenticated block for a new provider, you'll have to add it here too.
<details>
<summary><code>ProviderName</code> definition</summary>

```python title="backend/integrations/providers.py"
--8<-- "autogpt_platform/backend/backend/integrations/providers.py:ProviderName"
```
</details>

#### Multiple credentials inputs
Multiple credentials inputs are supported, under the following conditions:
- The name of each of the credentials input fields must end with `_credentials`.
- The names of the credentials input fields must match the names of the corresponding
  parameters on the `run(..)` method of the block.
- If more than one of the credentials parameters are required, `test_credentials`
  is a `dict[str, Credentials]`, with for each required credentials input the
  parameter name as the key and suitable test credentials as the value.


#### Adding an OAuth2 service integration

To add support for a new OAuth2-authenticated service, you'll need to add an `OAuthHandler`.
All our existing handlers and the base class can be found [here][OAuth2 handlers].

Every handler must implement the following parts of the [`BaseOAuthHandler`] interface:

```python title="backend/integrations/oauth/base.py"
--8<-- "autogpt_platform/backend/backend/integrations/oauth/base.py:BaseOAuthHandler1"
--8<-- "autogpt_platform/backend/backend/integrations/oauth/base.py:BaseOAuthHandler2"
--8<-- "autogpt_platform/backend/backend/integrations/oauth/base.py:BaseOAuthHandler3"
--8<-- "autogpt_platform/backend/backend/integrations/oauth/base.py:BaseOAuthHandler4"
--8<-- "autogpt_platform/backend/backend/integrations/oauth/base.py:BaseOAuthHandler5"
--8<-- "autogpt_platform/backend/backend/integrations/oauth/base.py:BaseOAuthHandler6"
```

As you can see, this is modeled after the standard OAuth2 flow.

Aside from implementing the `OAuthHandler` itself, adding a handler into the system requires two more things:

- Adding the handler class to `HANDLERS_BY_NAME` under [`integrations/oauth/__init__.py`](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform/backend/backend/integrations/oauth/__init__.py)

```python title="backend/integrations/oauth/__init__.py"
--8<-- "autogpt_platform/backend/backend/integrations/oauth/__init__.py:HANDLERS_BY_NAMEExample"
```

- Adding `{provider}_client_id` and `{provider}_client_secret` to the application's `Secrets` under [`util/settings.py`](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform/backend/backend/util/settings.py)

```python title="backend/util/settings.py"
--8<-- "autogpt_platform/backend/backend/util/settings.py:OAuthServerCredentialsExample"
```

[OAuth2 handlers]: https://github.com/Significant-Gravitas/AutoGPT/tree/master/autogpt_platform/backend/backend/integrations/oauth
[`BaseOAuthHandler`]: https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform/backend/backend/integrations/oauth/base.py

#### Adding to the frontend

You will need to add the provider (api or oauth) to the `CredentialsInput` component in [`frontend/src/components/integrations/credentials-input.tsx`](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform/frontend/src/components/integrations/credentials-input.tsx).

```ts title="frontend/src/components/integrations/credentials-input.tsx"
--8<-- "autogpt_platform/frontend/src/components/integrations/credentials-input.tsx:ProviderIconsEmbed"
```

You will also need to add the provider to the `CredentialsProvider` component in [`frontend/src/components/integrations/credentials-provider.tsx`](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform/frontend/src/components/integrations/credentials-provider.tsx).

```ts title="frontend/src/components/integrations/credentials-provider.tsx"
--8<-- "autogpt_platform/frontend/src/components/integrations/credentials-provider.tsx:CredentialsProviderNames"
```

Finally you will need to add the provider to the `CredentialsType` enum in [`frontend/src/lib/autogpt-server-api/types.ts`](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform/frontend/src/lib/autogpt-server-api/types.ts).

```ts title="frontend/src/lib/autogpt-server-api/types.ts"
--8<-- "autogpt_platform/frontend/src/lib/autogpt-server-api/types.ts:BlockIOCredentialsSubSchema"
```

#### Example: GitHub integration

- GitHub blocks with API key + OAuth2 support: [`blocks/github`](https://github.com/Significant-Gravitas/AutoGPT/tree/master/autogpt_platform/backend/backend/blocks/github/)

```python title="backend/blocks/github/issues.py"
--8<-- "autogpt_platform/backend/backend/blocks/github/issues.py:GithubCommentBlockExample"
```

- GitHub OAuth2 handler: [`integrations/oauth/github.py`](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform/backend/backend/integrations/oauth/github.py)

```python title="backend/integrations/oauth/github.py"
--8<-- "autogpt_platform/backend/backend/integrations/oauth/github.py:GithubOAuthHandlerExample"
```

#### Example: Google integration

- Google OAuth2 handler: [`integrations/oauth/google.py`](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform/backend/backend/integrations/oauth/google.py)

```python title="backend/integrations/oauth/google.py"
--8<-- "autogpt_platform/backend/backend/integrations/oauth/google.py:GoogleOAuthHandlerExample"
```

You can see that google has defined a `DEFAULT_SCOPES` variable, this is used to set the scopes that are requested no matter what the user asks for.

```python title="backend/blocks/google/_auth.py"
--8<-- "autogpt_platform/backend/backend/blocks/google/_auth.py:GoogleOAuthIsConfigured"
```

You can also see that `GOOGLE_OAUTH_IS_CONFIGURED` is used to disable the blocks that require OAuth if the oauth is not configured. This is in the `__init__` method of each block. This is because there is no api key fallback for google blocks so we need to make sure that the oauth is configured before we allow the user to use the blocks.

### Webhook-triggered Blocks

Webhook-triggered blocks allow your agent to respond to external events in real-time.
These blocks are triggered by incoming webhooks from third-party services
rather than being executed manually.

Creating and running a webhook-triggered block involves three main components:

- The block itself, which specifies:
    - Inputs for the user to select a resource and events to subscribe to
    - A `credentials` input with the scopes needed to manage webhooks
    - Logic to turn the webhook payload into outputs for the webhook block
- The `WebhooksManager` for the corresponding webhook service provider, which handles:
    - (De)registering webhooks with the provider
    - Parsing and validating incoming webhook payloads
- The credentials system for the corresponding service provider, which may include an `OAuthHandler`

There is more going on under the hood, e.g. to store and retrieve webhooks and their
links to nodes, but to add a webhook-triggered block you shouldn't need to make changes
to those parts of the system.

#### Creating a Webhook-triggered Block

To create a webhook-triggered block, follow these additional steps on top of the basic block creation process:

1. **Define `webhook_config`** in your block's `__init__` method.

    <details>
    <summary>Example: <code>GitHubPullRequestTriggerBlock</code></summary>

    ```python title="backend/blocks/github/triggers.py"
    --8<-- "autogpt_platform/backend/backend/blocks/github/triggers.py:example-webhook_config"
    ```
    </details>

    <details>
    <summary><code>BlockWebhookConfig</code> definition</summary>

    ```python title="backend/data/block.py"
    --8<-- "autogpt_platform/backend/backend/data/block.py:BlockWebhookConfig"
    ```
    </details>

2. **Define event filter input** in your block's Input schema.
    This allows the user to select which specific types of events will trigger the block in their agent.

    <details>
    <summary>Example: <code>GitHubPullRequestTriggerBlock</code></summary>

    ```python title="backend/blocks/github/triggers.py"
    --8<-- "autogpt_platform/backend/backend/blocks/github/triggers.py:example-event-filter"
    ```
    </details>

    - The name of the input field (`events` in this case) must match `webhook_config.event_filter_input`.
    - The event filter itself must be a Pydantic model with only boolean fields.

4. **Include payload field** in your block's Input schema.

    <details>
    <summary>Example: <code>GitHubTriggerBase</code></summary>

    ```python title="backend/blocks/github/triggers.py"
    --8<-- "autogpt_platform/backend/backend/blocks/github/triggers.py:example-payload-field"
    ```
    </details>

5. **Define `credentials` input** in your block's Input schema.
    - Its scopes must be sufficient to manage a user's webhooks through the provider's API
    - See [Blocks with authentication](#blocks-with-authentication) for further details

6. **Process webhook payload** and output relevant parts of it in your block's `run` method.

    <details>
    <summary>Example: <code>GitHubPullRequestTriggerBlock</code></summary>

    ```python
    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        yield "payload", input_data.payload
        yield "sender", input_data.payload["sender"]
        yield "event", input_data.payload["action"]
        yield "number", input_data.payload["number"]
        yield "pull_request", input_data.payload["pull_request"]
    ```

    Note that the `credentials` parameter can be omitted if the credentials
    aren't used at block runtime, like in the example.
    </details>

#### Adding a Webhooks Manager

To add support for a new webhook provider, you'll need to create a WebhooksManager that implements the `BaseWebhooksManager` interface:

```python title="backend/integrations/webhooks/_base.py"
--8<-- "autogpt_platform/backend/backend/integrations/webhooks/_base.py:BaseWebhooksManager1"

--8<-- "autogpt_platform/backend/backend/integrations/webhooks/_base.py:BaseWebhooksManager2"
--8<-- "autogpt_platform/backend/backend/integrations/webhooks/_base.py:BaseWebhooksManager3"
--8<-- "autogpt_platform/backend/backend/integrations/webhooks/_base.py:BaseWebhooksManager4"
--8<-- "autogpt_platform/backend/backend/integrations/webhooks/_base.py:BaseWebhooksManager5"
```

And add a reference to your `WebhooksManager` class in `WEBHOOK_MANAGERS_BY_NAME`:

```python title="backend/integrations/webhooks/__init__.py"
--8<-- "autogpt_platform/backend/backend/integrations/webhooks/__init__.py:WEBHOOK_MANAGERS_BY_NAME"
```

#### Example: GitHub Webhook Integration

<details>
<summary>
GitHub Webhook triggers: <a href="https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform/backend/backend/blocks/github/triggers.py"><code>blocks/github/triggers.py</code></a>
</summary>

```python title="backend/blocks/github/triggers.py"
--8<-- "autogpt_platform/backend/backend/blocks/github/triggers.py:GithubTriggerExample"
```
</details>

<details>
<summary>
GitHub Webhooks Manager: <a href="https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform/backend/backend/integrations/webhooks/github.py"><code>integrations/webhooks/github.py</code></a>
</summary>

```python title="backend/integrations/webhooks/github.py"
--8<-- "autogpt_platform/backend/backend/integrations/webhooks/github.py:GithubWebhooksManager"
```
</details>

## Key Points to Remember

- **Unique ID**: Give your block a unique ID in the **init** method.
- **Input and Output Schemas**: Define clear input and output schemas.
- **Error Handling**: Implement error handling in the `run` method.
- **Output Results**: Use `yield` to output results in the `run` method.
- **Testing**: Provide test input and output in the **init** method for automatic testing.

## Understanding the Testing Process

The testing of blocks is handled by `test_block.py`, which does the following:

1. It calls the block with the provided `test_input`.
   If the block has a `credentials` field, `test_credentials` is passed in as well.
2. If a `test_mock` is provided, it temporarily replaces the specified methods with the mock functions.
3. It then asserts that the output matches the `test_output`.

For the WikipediaSummaryBlock:

- The test will call the block with the topic "Artificial Intelligence".
- Instead of making a real API call, it will use the mock function, which returns `{"extract": "summary content"}`.
- It will then check if the output key is "summary" and its value is a string.

This approach allows us to test the block's logic comprehensively without relying on external services, while also accommodating non-deterministic outputs.

## Security Best Practices for SSRF Prevention

When creating blocks that handle external URL inputs or make network requests, it's crucial to use the platform's built-in SSRF protection mechanisms. The `backend.util.request` module provides a secure `Requests` wrapper class that should be used for all HTTP requests.

### Using the Secure Requests Wrapper

```python
from backend.util.request import requests

class MyNetworkBlock(Block):
    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            # The requests wrapper automatically validates URLs and blocks dangerous requests
            response = requests.get(input_data.url)
            yield "result", response.text
        except ValueError as e:
            # URL validation failed
            raise RuntimeError(f"Invalid URL provided: {e}")
        except requests.exceptions.RequestException as e:
            # Request failed
            raise RuntimeError(f"Request failed: {e}")
```

The `Requests` wrapper provides these security features:

1. **URL Validation**:
    - Blocks requests to private IP ranges (RFC 1918)
    - Validates URL format and protocol
    - Resolves DNS and checks IP addresses
    - Supports whitelisting trusted origins

2. **Secure Defaults**:
    - Disables redirects by default
    - Raises exceptions for non-200 status codes
    - Supports custom headers and validators

3. **Protected IP Ranges**:
   The wrapper denies requests to these networks:

    ```python title="backend/util/request.py"
    --8<-- "autogpt_platform/backend/backend/util/request.py:BLOCKED_IP_NETWORKS"
    ```

### Custom Request Configuration

If you need to customize the request behavior:

```python
from backend.util.request import Requests

# Create a custom requests instance with specific trusted origins
custom_requests = Requests(
    trusted_origins=["api.trusted-service.com"],
    raise_for_status=True,
    extra_headers={"User-Agent": "MyBlock/1.0"}
)
```

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

## Blocks we want to see

Below is a list of blocks that we would like to see implemented in the AutoGPT Agent Server. If you're interested in contributing, feel free to pick one of these blocks or chose your own.

If you would like to implement one of these blocks, open a pull request and we will start the review process.

### Consumer Services/Platforms

- Google sheets - [~~Read/Append~~](https://github.com/Significant-Gravitas/AutoGPT/pull/8236)
- Email - Read/Send with [~~Gmail~~](https://github.com/Significant-Gravitas/AutoGPT/pull/8236), Outlook, Yahoo, Proton, etc
- Calendar - Read/Write with Google Calendar, Outlook Calendar, etc
- Home Assistant - Call Service, Get Status
- Dominos - Order Pizza, Track Order
- Uber - Book Ride, Track Ride
- Notion - Create/Read Page, Create/Append/Read DB
- Google drive - read/write/overwrite file/folder

### Social Media

- Twitter - Post, Reply, Get Replies, Get Comments, Get Followers, Get Following, Get Tweets, Get Mentions
- Instagram - Post, Reply, Get Comments, Get Followers, Get Following, Get Posts, Get Mentions, Get Trending Posts
- TikTok - Post, Reply, Get Comments, Get Followers, Get Following, Get Videos, Get Mentions, Get Trending Videos
- LinkedIn - Post, Reply, Get Comments, Get Followers, Get Following, Get Posts, Get Mentions, Get Trending Posts
- YouTube - Transcribe Videos/Shorts, Post Videos/Shorts, Read/Reply/React to Comments, Update Thumbnails, Update Description, Update Tags, Update Titles, Get Views, Get Likes, Get Dislikes, Get Subscribers, Get Comments, Get Shares, Get Watch Time, Get Revenue, Get Trending Videos, Get Top Videos, Get Top Channels
- Reddit - Post, Reply, Get Comments, Get Followers, Get Following, Get Posts, Get Mentions, Get Trending Posts
- Treatwell (and related Platforms) - Book, Cancel, Review, Get Recommendations
- Substack - Read/Subscribe/Unsubscribe, Post/Reply, Get Recommendations
- Discord - Read/Post/Reply, Moderation actions
- GoodReads - Read/Post/Reply, Get Recommendations

### E-commerce

- Airbnb - Book, Cancel, Review, Get Recommendations
- Amazon - Order, Track Order, Return, Review, Get Recommendations
- eBay - Order, Track Order, Return, Review, Get Recommendations
- Upwork - Post Jobs, Hire Freelancer, Review Freelancer, Fire Freelancer

### Business Tools

- External Agents - Call other agents similar to AutoGPT
- Trello - Create/Read/Update/Delete Cards, Lists, Boards
- Jira - Create/Read/Update/Delete Issues, Projects, Boards
- Linear - Create/Read/Update/Delete Issues, Projects, Boards
- Excel - Read/Write/Update/Delete Rows, Columns, Sheets
- Slack - Read/Post/Reply to Messages, Create Channels, Invite Users
- ERPNext - Create/Read/Update/Delete Invoices, Orders, Customers, Products
- Salesforce - Create/Read/Update/Delete Leads, Opportunities, Accounts
- HubSpot - Create/Read/Update/Delete Contacts, Deals, Companies
- Zendesk - Create/Read/Update/Delete Tickets, Users, Organizations
- Odoo - Create/Read/Update/Delete Sales Orders, Invoices, Customers
- Shopify - Create/Read/Update/Delete Products, Orders, Customers
- WooCommerce - Create/Read/Update/Delete Products, Orders, Customers
- Squarespace - Create/Read/Update/Delete Pages, Products, Orders

## Agent Templates we want to see

### Data/Information

- Summarize top news of today, of this week, this month via Apple News or other large media outlets BBC, TechCrunch, hackernews, etc
- Create, read, and summarize substack newsletters or any newsletters (blog writer vs blog reader)
- Get/read/summarize the most viral Twitter, Instagram, TikTok (general social media accounts) of the day, week, month
- Get/Read any LinkedIn posts or profile that mention AI Agents
- Read/Summarize discord (might not be able to do this because you need access)
- Read / Get most read books in a given month, year, etc from GoodReads or Amazon Books, etc
- Get dates for specific shows across all streaming services
  - Suggest/Recommend/Get most watched shows in a given month, year, etc across all streaming platforms
- Data analysis from xlsx data set
  - Gather via Excel or Google Sheets data > Sample the data randomly (sample block takes top X, bottom X, randomly, etc) > pass that to LLM Block to generate a script for analysis of the full data > Python block to run the script> making a loop back through LLM Fix Block on error > create chart/visualization (potentially in the code block?) > show the image as output (this may require frontend changes to show)
- Tiktok video search and download

### Marketing

- Portfolio site design and enhancements
