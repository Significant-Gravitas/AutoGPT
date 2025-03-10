# Tutorial: Adding a New Integration Block to the AutoGPT Platform

In this tutorial, we’ll walk through the process of adding a new integration block to the AutoGPT Platform, using a fictional "Example API" as our case study. This guide ensures every step is clear, from defining the block to integrating it across both backend and frontend files. By the end, you’ll have a fully functional integration complete with credentials, API interactions, webhook support, and a polished user interface. Let’s dive in.

---

## Understanding Blocks in the AutoGPT Platform

Before we start coding, it’s worth understanding what a block is in the AutoGPT Platform. A block is a modular, reusable unit of functionality that powers workflows. There are two main types: standard blocks, which users can execute manually or call within workflows to perform tasks like API requests, and trigger blocks, which activate automatically in response to external events, such as webhooks. For this tutorial, we’ll create both an `ExampleBlock` (a standard block) and an `ExampleTriggerBlock` (a trigger block) to showcase the full scope of integration.

To keep our blocks effective, we’ll design them with modularity in mind, focusing each on a single task. We’ll also ensure they’re reusable across workflows, handle errors gracefully, and include test cases for validation. With that foundation, let’s move on to setting up our file structure.

---

## File Layout for the Integration

Adding a new integration requires updates across multiple files in the AutoGPT Platform’s backend and frontend. Here’s the complete layout we’ll work with:

### Backend Files
The backend handles the core logic, credentials, and server-side interactions. We’ll modify or create the following files:

- `backend/.env.example`: Adds an environment variable for the Example API key.
- `backend/blocks/example/_api.py`: Defines the API client for interacting with the Example API.
- `backend/blocks/example/_auth.py`: Sets up credentials and test credentials for authentication.
- `backend/blocks/example/example.py`: Implements the standard block (`ExampleBlock`).
- `backend/blocks/example/triggers.py`: Implements the trigger block (`ExampleTriggerBlock`).
- `backend/data/block_cost_config.py`: Tracks usage costs for the block.
- `backend/integrations/credentials_store.py`: Registers Example API credentials in the platform’s credential store.
- `backend/integrations/providers.py`: Defines the provider name for the Example API.
- `backend/integrations/webhooks/example.py`: Manages webhook functionality.
- `backend/util/settings.py`: Adds the API key to the platform’s secrets model.

### Frontend Files
The frontend ensures users can interact with the integration through the UI. We’ll update these files:

- `frontend/src/components/integrations/credentials-input.tsx`: Adds an icon for the Example API provider.
- `frontend/src/components/integrations/credentials-provider.tsx`: Sets the display name for the provider in the UI.
- `frontend/src/lib/autogpt-server-api/types.ts`: Registers the provider name as a constant for API interactions.

This structure keeps everything organized and ensures the integration spans both backend logic and frontend presentation. With the layout in place, let’s start building.

---

## Setting Up a Credential Provider

Credentials are essential for authenticating our blocks with the Example API. Here’s how we set them up across multiple files.

### Defining the Provider Name
First, we need to register the Example API as a recognized provider in the platform. In `backend/integrations/providers.py`, add this line:

```python
EXAMPLE_PROVIDER = "example-provider"
```

This addition to the `ProviderName` enum uniquely identifies the Example API, allowing the platform to link credentials and blocks to it.

### Defining Credentials Input and Test Credentials
Next, in `backend/blocks/example/_auth.py`, we define how credentials are structured and provide a mock credential for testing:

```python
ExampleCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.EXAMPLE_PROVIDER], Literal["api_key"]
]
```

This code creates `ExampleCredentialsInput`, specifying that it’s tied to the "example-provider" and expects an API key. The use of `Literal` ensures type safety, making it clear what credential type is required.

For testing, we also define:

```python
TEST_CREDENTIALS = APIKeyCredentials(
    id="9191c4f0-498f-4235-a79c-59c0e37454d4",
    provider="example-provider",
    api_key=SecretStr("mock-example-api-key"),
    title="Mock Example API key",
    expires_at=None,
)
```

This mock credential includes a unique ID, the provider name, a secure API key (hidden via `SecretStr`), a title, and no expiration. It’s perfect for testing without needing a real key.

### Updating the Credentials Store
To make real credentials available, update `backend/integrations/credentials_store.py`:

```python
example_credentials = APIKeyCredentials(
    id="a2b7f68f-aa6a-4995-99ec-b45b40d33498",
    provider="example-provider",
    api_key=SecretStr(settings.secrets.example_api_key),
    title="Use Credits for Example",
    expires_at=None,
)
DEFAULT_CREDENTIALS.append(example_credentials)
```

This code creates a live credential using an API key from the environment, adds it to the platform’s credential list, and ensures it’s accessible to blocks.

### Configuring Environment Variables
In `backend/.env.example`, add:

```
EXAMPLE_API_KEY=
```

This line prompts users to provide their API key in their `.env` file. Then, in `backend/util/settings.py`, update the `Secrets` model:

```python
example_api_key: str = Field(default="", description="Example API Key")
```

This addition loads the key from the environment, keeping it secure and optional with a default empty value.

---

## Creating an API Client

The API client, defined in `backend/blocks/example/_api.py`, handles communication with the Example API. Here’s how we build it.

### Initializing the Client
Start with:

```python
class ExampleClient:
    API_BASE_URL = "https://api.example.com/v1"

    def __init__(self, credentials: Optional[APIKeyCredentials] = None, custom_requests: Optional[Requests] = None):
        headers = {"Content-Type": "application/json"}
        if credentials:
            headers["Authorization"] = credentials.auth_header()
        self._requests = custom_requests or Requests(extra_headers=headers, trusted_origins=["https://api.example.com"])
```

This class sets a base URL and initializes with optional credentials and a custom request handler (useful for testing). It builds headers with a JSON content type and, if credentials are provided, an authorization header. The `Requests` utility ensures secure, standardized HTTP requests.

### Implementing API Methods
Add a method to create resources:

```python
def create_resource(self, data: Dict) -> Dict:
    try:
        response = self._requests.post(f"{self.API_BASE_URL}/resources", json=data)
        return self._handle_response(response)
    except Exception as e:
        raise ExampleAPIException(f"Failed to create resource: {str(e)}", 500)
```

This method sends a POST request with data (e.g., `{"name": "Hello"}`), processes the response, and handles errors with a custom exception, ensuring robust error reporting.

---

## Building the Standard Block

In `backend/blocks/example/example.py`, we define `ExampleBlock`:

### Defining Input and Output Schemas
Start with:

```python
class ExampleBlock(Block):
    class Input(BlockSchema):
        name: str = SchemaField(description="The name of the example block")
        greetings: list[str] = SchemaField(default=["Hello", "Hi", "Hey"])
        is_funny: bool = SchemaField(default=True)
        credentials: ExampleCredentialsInput = CredentialsField()

    class Output(BlockSchema):
        response: dict[str, Any] = SchemaField()
        all_responses: list[dict[str, Any]] = SchemaField()
        greeting_count: int = SchemaField()
        error: str = SchemaField()
```

The `Input` schema accepts a name, a list of greetings, a boolean, and credentials, while `Output` defines what the block returns: individual responses, all responses, a greeting count, and an error message.

### Implementing the Run Method
Add:

```python
def run(self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs) -> BlockOutput:
    rtn_all_responses = []
    for greeting in input_data.greetings:
        full_greeting = self.my_static_method(greeting)
        message = self.my_function_that_can_be_mocked(full_greeting, credentials)
        rtn_all_responses.append({"message": message})
        yield "response", {"message": message}
    yield "all_responses", rtn_all_responses
    yield "greeting_count", len(input_data.greetings)
```

This method loops through greetings, processes them, calls the API, and yields results incrementally. The `my_function_that_can_be_mocked` method, defined as:

```python
def my_function_that_can_be_mocked(self, input: str, credentials: APIKeyCredentials) -> str:
    client = ExampleClient(credentials=credentials)
    resource_data = {"name": input, "type": "greeting"}
    response = client.create_resource(resource_data)
    return f"API response: {response.get('message', 'Hello, world!')}"
```

This connects to the API client, sends data, and formats the response.

---

## Managing Webhooks

In `backend/integrations/webhooks/example.py`, set up webhook support:

### Defining Event Types
Define:

```python
class ExampleWebhookEventType(StrEnum):
    EXAMPLE_EVENT = "example_event"
    ANOTHER_EXAMPLE_EVENT = "another_example_event"
```

This enum lists possible webhook events from the Example API.

### Creating the Webhook Manager
Add:

```python
class ExampleWebhookManager(ManualWebhookManagerBase):
    PROVIDER_NAME = ProviderName.EXAMPLE_PROVIDER
    WebhookEventType = ExampleWebhookEventType
    BASE_URL = "https://api.example.com"
```

This class ties the webhook manager to the Example API provider and event types.

### Validating Payloads
Implement:

```python
async def validate_payload(cls, webhook: integrations.Webhook, request: Request) -> tuple[dict, str]:
    payload = await request.json()
    event_type = payload.get("webhook_type", ExampleWebhookEventType.EXAMPLE_EVENT)
    return payload, event_type
```

This async method extracts and validates webhook payloads, defaulting to `"example_event"`.

---

## Adding a Trigger Block

In `backend/blocks/example/triggers.py`, define `ExampleTriggerBlock`:

```python
class ExampleTriggerBlock(Block):
    class Input(BlockSchema):
        payload: dict = SchemaField(hidden=True)

    class Output(BlockSchema):
        event_data: dict = SchemaField(description="The contents of the example webhook event.")

    webhook_config = BlockManualWebhookConfig(
        provider="example_provider",
        webhook_type=ExampleWebhookEventType.EXAMPLE_EVENT,
    )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        logger.info("Example trigger block run with payload: %s", input_data.payload)
        yield "event_data", input_data.payload
```

This block takes a hidden payload, outputs event data, links to the webhook config, and logs and yields the payload when triggered.

---

## Updating the Frontend

Finally, integrate the provider into the frontend.

### Adding a Provider Icon
In `frontend/src/components/integrations/credentials-input.tsx`, add:

```typescript
example: fallbackIcon
```

This assigns a fallback icon to the "example" provider.

### Setting the Display Name
In `frontend/src/components/integrations/credentials-provider.tsx`, add:

```typescript
example: "Example"
```

This sets "Example" as the UI display name.

### Registering the Provider Constant
In `frontend/src/lib/autogpt-server-api/types.ts`, add:

```typescript
EXAMPLE: "example"
```

This ensures the provider is recognized in API interactions.

---

## Final Steps

To wrap up, update `backend/data/block_cost_config.py` with a cost entry for `ExampleBlock`, test both blocks with sample inputs, and document their usage in the platform’s docs. You’ve now added a complete integration to the AutoGPT Platform, spanning backend logic, API calls, webhooks, and a user-friendly frontend. Happy coding!
