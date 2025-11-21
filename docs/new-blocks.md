# Contributing to AutoGPT Agent Server: Creating and Testing Blocks

This guide will walk you through the process of creating and testing a new block for the AutoGPT Agent Server, using the WikipediaSummaryBlock as an example.

{% hint style="success" %}
`New SDK-Based Approach`
{% endhint %}

For a more comprehensive guide using the new SDK pattern with ProviderBuilder and advanced features like OAuth and webhooks, see the [Block SDK Guide](https://docs.agpt.co/platform/block-sdk-guide/).

### Understanding Blocks and Testing <a href="#understanding-blocks-and-testing" id="understanding-blocks-and-testing"></a>

Blocks are reusable components that can be connected to form a graph representing an agent's behavior. Each block has inputs, outputs, and a specific function. Proper testing is crucial to ensure blocks work correctly and consistently.

### Creating and Testing a New Block <a href="#creating-and-testing-a-new-block" id="creating-and-testing-a-new-block"></a>

Follow these steps to create and test a new block:

1. **Create a new Python file** for your block in the `autogpt_platform/backend/backend/blocks` directory. Name it descriptively and use snake\_case. For example: `get_wikipedia_summary.py`.
2. **Import necessary modules and create a class that inherits from `Block`**. Make sure to include all necessary imports for your block.

Every block should contain the following:

```
from backend.data.block import Block, BlockSchemaInput, BlockSchemaOutput, BlockOutput
```

Example for the Wikipedia summary block:

```
from backend.data.block import Block, BlockSchemaInput, BlockSchemaOutput, BlockOutput
from backend.utils.get_request import GetRequest
import requests

class WikipediaSummaryBlock(Block, GetRequest):
    # Block implementation will go here
```

1. **Define the input and output schemas** using `BlockSchema`. These schemas specify the data structure that the block expects to receive (input) and produce (output).
2. The input schema defines the structure of the data the block will process. Each field in the schema represents a required piece of input data.
3. The output schema defines the structure of the data the block will return after processing. Each field in the schema represents a piece of output data.

Example:

```
class Input(BlockSchemaInput):
    topic: str  # The topic to get the Wikipedia summary for

class Output(BlockSchemaOutput):
    summary: str  # The summary of the topic from Wikipedia
```

1. **Implement the `__init__` method, including test data and mocks:**

!!! important Use UUID generator (e.g. https://www.uuidgenerator.net/) for every new block `id` and _do not_ make up your own. Alternatively, you can run this python code to generate an uuid: `print(__import__('uuid').uuid4())`

```
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

* `id`: A unique identifier for the block.
* `input_schema` and `output_schema`: Define the structure of the input and output data.

Let's break down the testing components:

* `test_input`: This is a sample input that will be used to test the block. It should be a valid input according to your Input schema.
* `test_output`: This is the expected output when running the block with the `test_input`. It should match your Output schema. For non-deterministic outputs or when you only want to assert the type, you can use Python types instead of specific values. In this example, `("summary", str)` asserts that the output key is "summary" and its value is a string.
* `test_mock`: This is crucial for blocks that make network calls. It provides a mock function that replaces the actual network call during testing.

In this case, we're mocking the `get_request` method to always return a dictionary with an 'extract' key, simulating a successful API response. This allows us to test the block's logic without making actual network requests, which could be slow, unreliable, or rate-limited.

1. **Implement the `run` method with error handling.** This should contain the main logic of the block:

```
def run(self, input_data: Input, **kwargs) -> BlockOutput:
    try:
        topic = input_data.topic
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"

        response = self.get_request(url, json=True)
        yield "summary", response['extract']

    except requests.exceptions.HTTPError as http_err:
        raise RuntimeError(f"HTTP error occurred: {http_err}")
```

* **Try block**: Contains the main logic to fetch and process the Wikipedia summary.
* **API request**: Send a GET request to the Wikipedia API.
* **Error handling**: Handle various exceptions that might occur during the API request and data processing. We don't need to catch all exceptions, only the ones we expect and can handle. The uncaught exceptions will be automatically yielded as `error` in the output. Any block that raises an exception (or yields an `error` output) will be marked as failed. Prefer raising exceptions over yielding `error`, as it will stop the execution immediately.
* **Yield**: Use `yield` to output the results. Prefer to output one result object at a time. If you are calling a function that returns a list, you can yield each item in the list separately. You can also yield the whole list as well, but do both rather than yielding the list. For example: If you were writing a block that outputs emails, you'd yield each email as a separate result object, but you could also yield the whole list as an additional single result object. Yielding output named `error` will break the execution right away and mark the block execution as failed.
* **kwargs**: The `kwargs` parameter is used to pass additional arguments to the block. It is not used in the example above, but it is available to the block. You can also have args as inline signatures in the run method ala `def run(self, input_data: Input, *, user_id: str, **kwargs) -> BlockOutput:`. Available kwargs are:
  * `user_id`: The ID of the user running the block.
  * `graph_id`: The ID of the agent that is executing the block. This is the same for every version of the agent
  * `graph_exec_id`: The ID of the execution of the agent. This changes every time the agent has a new "run"
  * `node_exec_id`: The ID of the execution of the node. This changes every time the node is executed
  * `node_id`: The ID of the node that is being executed. It changes every version of the graph, but not every time the node is executed.

#### Field Types <a href="#field-types" id="field-types"></a>

**OneOf Fields**

`oneOf` allows you to specify that a field must be exactly one of several possible options. This is useful when you want your block to accept different types of inputs that are mutually exclusive.

Example:

```
attachment: Union[Media, DeepLink, Poll, Place, Quote] = SchemaField(
    discriminator='discriminator',
    description="Attach either media, deep link, poll, place or quote - only one can be used"
)
```

The `discriminator` parameter tells AutoGPT which field to look at in the input to determine which type it is.

In each model, you need to define the discriminator value:

```
class Media(BaseModel):
    discriminator: Literal['media']
    media_ids: List[str]

class DeepLink(BaseModel):
    discriminator: Literal['deep_link']
    direct_message_deep_link: str
```

**OptionalOneOf Fields**

`OptionalOneOf` is similar to `oneOf` but allows the field to be optional (None). This means the field can be either one of the specified types or None.

Example:

```
attachment: Union[Media, DeepLink, Poll, Place, Quote] | None = SchemaField(
    discriminator='discriminator',
    description="Optional attachment - can be media, deep link, poll, place, quote or None"
)
```

The key difference is the `| None` which makes the entire field optional.

#### Blocks with Authentication <a href="#blocks-with-authentication" id="blocks-with-authentication"></a>

Our system supports auth offloading for API keys and OAuth2 authorization flows. Adding a block with API key authentication is straight-forward, as is adding a block for a service that we already have OAuth2 support for.

Implementing the block itself is relatively simple. On top of the instructions above, you're going to add a `credentials` parameter to the `Input` model and the `run` method:

```
from backend.data.model import (
    APIKeyCredentials,
    OAuth2Credentials,
    Credentials,
)

from backend.data.block import Block, BlockOutput, BlockSchemaInput, BlockSchemaOutput
from backend.data.model import CredentialsField
from backend.integrations.providers import ProviderName


# API Key auth:
class BlockWithAPIKeyAuth(Block):
    class Input(BlockSchemaInput):
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
    class Input(BlockSchemaInput):
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
    class Input(BlockSchemaInput):
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

The `APIKeyCredentials` and `OAuth2Credentials` models are defined [here](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform/autogpt_libs/autogpt_libs/supabase_integration_credentials_store/types.py). To use them in e.g. an API request, you can either access the token directly:

```
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

```
# credentials: APIKeyCredentials | OAuth2Credentials
response = requests.post(
    url,
    headers={"Authorization": credentials.auth_header()},
)
```

The `ProviderName` enum is the single source of truth for which providers exist in our system. Naturally, to add an authenticated block for a new provider, you'll have to add it here too.

<details>

<summary><code>ProviderName</code> definition</summary>

backend/integrations/providers.py

```
class ProviderName(str, Enum):
    """
    Provider names for integrations.

    This enum extends str to accept any string value while maintaining
    backward compatibility with existing provider constants.
    """

    AIML_API = "aiml_api"
    ANTHROPIC = "anthropic"
    APOLLO = "apollo"
    COMPASS = "compass"
    DISCORD = "discord"
    D_ID = "d_id"
    E2B = "e2b"
    FAL = "fal"
    GITHUB = "github"
    GOOGLE = "google"
    GOOGLE_MAPS = "google_maps"
    GROQ = "groq"
    HTTP = "http"
    HUBSPOT = "hubspot"
    ENRICHLAYER = "enrichlayer"
    IDEOGRAM = "ideogram"
    JINA = "jina"
    LLAMA_API = "llama_api"
    MEDIUM = "medium"
    MEM0 = "mem0"
    NOTION = "notion"
    NVIDIA = "nvidia"
    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENWEATHERMAP = "openweathermap"
    OPEN_ROUTER = "open_router"
    PINECONE = "pinecone"
    REDDIT = "reddit"
    REPLICATE = "replicate"
    REVID = "revid"
    SCREENSHOTONE = "screenshotone"
    SLANT3D = "slant3d"
    SMARTLEAD = "smartlead"
    SMTP = "smtp"
    TWITTER = "twitter"
    TODOIST = "todoist"
    UNREAL_SPEECH = "unreal_speech"
    V0 = "v0"
    ZEROBOUNCE = "zerobounce"

    @classmethod
    def _missing_(cls, value: Any) -> "ProviderName":
        """
        Allow any string value to be used as a ProviderName.
        This enables SDK users to define custom providers without
        modifying the enum.
        """
        if isinstance(value, str):
            # Create a pseudo-member that behaves like an enum member
            pseudo_member = str.__new__(cls, value)
            pseudo_member._name_ = value.upper()
            pseudo_member._value_ = value
            return pseudo_member
        return None  # type: ignore

    @classmethod
    def __get_pydantic_json_schema__(cls, schema, handler):
        """
        Custom JSON schema generation that allows any string value,
        not just the predefined enum values.
        """
        # Get the default schema
        json_schema = handler(schema)

        # Remove the enum constraint to allow any string
        if "enum" in json_schema:
            del json_schema["enum"]

        # Keep the type as string
        json_schema["type"] = "string"

        # Update description to indicate custom providers are allowed
        json_schema["description"] = (
            "Provider name for integrations. "
            "Can be any string value, including custom provider names."
        )

        return json_schema

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """
        Pydantic v2 core schema that allows any string value.
        """
        from pydantic_core import core_schema

        # Create a string schema that validates any string
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.str_schema(),
        )
```

</details>

**Multiple Credentials Inputs**

Multiple credentials inputs are supported, under the following conditions:

* The name of each of the credentials input fields must end with `_credentials`.
* The names of the credentials input fields must match the names of the corresponding parameters on the `run(..)` method of the block.
* If more than one of the credentials parameters are required, `test_credentials` is a `dict[str, Credentials]`, with for each required credentials input the parameter name as the key and suitable test credentials as the value.

**Adding an OAuth2 Service Integration**

To add support for a new OAuth2-authenticated service, you'll need to add an `OAuthHandler`. All our existing handlers and the base class can be found [here](https://github.com/Significant-Gravitas/AutoGPT/tree/master/autogpt_platform/backend/backend/integrations/oauth).

Every handler must implement the following parts of the \[`BaseOAuthHandler`] interface:

backend/integrations/oauth/base.py

```
PROVIDER_NAME: ClassVar[ProviderName | str]
DEFAULT_SCOPES: ClassVar[list[str]] = []
def __init__(self, client_id: str, client_secret: str, redirect_uri: str): ...

def get_login_url(
    self, scopes: list[str], state: str, code_challenge: Optional[str]
) -> str:
async def exchange_code_for_tokens(
    self, code: str, scopes: list[str], code_verifier: Optional[str]
) -> OAuth2Credentials:
async def _refresh_tokens(
    self, credentials: OAuth2Credentials
) -> OAuth2Credentials:
async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
```

As you can see, this is modeled after the standard OAuth2 flow.

Aside from implementing the `OAuthHandler` itself, adding a handler into the system requires two more things:

* Adding the handler class to `HANDLERS_BY_NAME` under [`integrations/oauth/__init__.py`](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform/backend/backend/integrations/oauth/__init__.py)

backend/integrations/oauth/\_\_init\_\_.py

```
# Build handlers dict with string keys for compatibility with SDK auto-registration
_ORIGINAL_HANDLERS = [
    DiscordOAuthHandler,
    GitHubOAuthHandler,
    GoogleOAuthHandler,
    NotionOAuthHandler,
    TwitterOAuthHandler,
    TodoistOAuthHandler,
]

# Start with original handlers
_handlers_dict = {
    (
        handler.PROVIDER_NAME.value
        if hasattr(handler.PROVIDER_NAME, "value")
        else str(handler.PROVIDER_NAME)
    ): handler
    for handler in _ORIGINAL_HANDLERS
}


class SDKAwareCredentials(BaseModel):
    """OAuth credentials configuration."""

    use_secrets: bool = True
    client_id_env_var: Optional[str] = None
    client_secret_env_var: Optional[str] = None


_credentials_by_provider = {}
# Add default credentials for original handlers
for handler in _ORIGINAL_HANDLERS:
    provider_name = (
        handler.PROVIDER_NAME.value
        if hasattr(handler.PROVIDER_NAME, "value")
        else str(handler.PROVIDER_NAME)
    )
    _credentials_by_provider[provider_name] = SDKAwareCredentials(
        use_secrets=True, client_id_env_var=None, client_secret_env_var=None
    )


# Create a custom dict class that includes SDK handlers
class SDKAwareHandlersDict(dict):
    """Dictionary that automatically includes SDK-registered OAuth handlers."""

    def __getitem__(self, key):
        # First try the original handlers
        if key in _handlers_dict:
            return _handlers_dict[key]

        # Then try SDK handlers
        try:
            from backend.sdk import AutoRegistry

            sdk_handlers = AutoRegistry.get_oauth_handlers()
            if key in sdk_handlers:
                return sdk_handlers[key]
        except ImportError:
            pass

        # If not found, raise KeyError
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        if key in _handlers_dict:
            return True
        try:
            from backend.sdk import AutoRegistry

            sdk_handlers = AutoRegistry.get_oauth_handlers()
            return key in sdk_handlers
        except ImportError:
            return False

    def keys(self):
        # Combine all keys into a single dict and return its keys view
        combined = dict(_handlers_dict)
        try:
            from backend.sdk import AutoRegistry

            sdk_handlers = AutoRegistry.get_oauth_handlers()
            combined.update(sdk_handlers)
        except ImportError:
            pass
        return combined.keys()

    def values(self):
        combined = dict(_handlers_dict)
        try:
            from backend.sdk import AutoRegistry

            sdk_handlers = AutoRegistry.get_oauth_handlers()
            combined.update(sdk_handlers)
        except ImportError:
            pass
        return combined.values()

    def items(self):
        combined = dict(_handlers_dict)
        try:
            from backend.sdk import AutoRegistry

            sdk_handlers = AutoRegistry.get_oauth_handlers()
            combined.update(sdk_handlers)
        except ImportError:
            pass
        return combined.items()


class SDKAwareCredentialsDict(dict):
    """Dictionary that automatically includes SDK-registered OAuth credentials."""

    def __getitem__(self, key):
        # First try the original handlers
        if key in _credentials_by_provider:
            return _credentials_by_provider[key]

        # Then try SDK credentials
        try:
            from backend.sdk import AutoRegistry

            sdk_credentials = AutoRegistry.get_oauth_credentials()
            if key in sdk_credentials:
                # Convert from SDKOAuthCredentials to SDKAwareCredentials
                sdk_cred = sdk_credentials[key]
                return SDKAwareCredentials(
                    use_secrets=sdk_cred.use_secrets,
                    client_id_env_var=sdk_cred.client_id_env_var,
                    client_secret_env_var=sdk_cred.client_secret_env_var,
                )
        except ImportError:
            pass

        # If not found, raise KeyError
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        if key in _credentials_by_provider:
            return True
        try:
            from backend.sdk import AutoRegistry

            sdk_credentials = AutoRegistry.get_oauth_credentials()
            return key in sdk_credentials
        except ImportError:
            return False

    def keys(self):
        # Combine all keys into a single dict and return its keys view
        combined = dict(_credentials_by_provider)
        try:
            from backend.sdk import AutoRegistry

            sdk_credentials = AutoRegistry.get_oauth_credentials()
            combined.update(sdk_credentials)
        except ImportError:
            pass
        return combined.keys()

    def values(self):
        combined = dict(_credentials_by_provider)
        try:
            from backend.sdk import AutoRegistry

            sdk_credentials = AutoRegistry.get_oauth_credentials()
            # Convert SDK credentials to SDKAwareCredentials
            for key, sdk_cred in sdk_credentials.items():
                combined[key] = SDKAwareCredentials(
                    use_secrets=sdk_cred.use_secrets,
                    client_id_env_var=sdk_cred.client_id_env_var,
                    client_secret_env_var=sdk_cred.client_secret_env_var,
                )
        except ImportError:
            pass
        return combined.values()

    def items(self):
        combined = dict(_credentials_by_provider)
        try:
            from backend.sdk import AutoRegistry

            sdk_credentials = AutoRegistry.get_oauth_credentials()
            # Convert SDK credentials to SDKAwareCredentials
            for key, sdk_cred in sdk_credentials.items():
                combined[key] = SDKAwareCredentials(
                    use_secrets=sdk_cred.use_secrets,
                    client_id_env_var=sdk_cred.client_id_env_var,
                    client_secret_env_var=sdk_cred.client_secret_env_var,
                )
        except ImportError:
            pass
        return combined.items()


HANDLERS_BY_NAME: dict[str, type["BaseOAuthHandler"]] = SDKAwareHandlersDict()
CREDENTIALS_BY_PROVIDER: dict[str, SDKAwareCredentials] = SDKAwareCredentialsDict()
```

* Adding `{provider}_client_id` and `{provider}_client_secret` to the application's `Secrets` under [`util/settings.py`](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform/backend/backend/util/settings.py)

backend/util/settings.py

```
github_client_id: str = Field(default="", description="GitHub OAuth client ID")
github_client_secret: str = Field(
    default="", description="GitHub OAuth client secret"
)
```

**Adding to the Frontend**

You will need to add the provider (api or oauth) to the `CredentialsInput` component in [`/frontend/src/app/(platform)/library/agents/[id]/components/AgentRunsView/components/CredentialsInputs/CredentialsInputs.tsx`](https://github.com/Significant-Gravitas/AutoGPT/blob/dev/autogpt_platform/frontend/src/app/\(platform\)/library/agents/\[id]/components/AgentRunsView/components/CredentialsInputs/CredentialsInputs.tsx).

frontend/src/components/integrations/credentials-input.tsx

```
--8 <
  --"autogpt_platform/frontend/src/app/(platform)/library/agents/[id]/components/AgentRunsView/components/CredentialsInputs/CredentialsInputs.tsx:ProviderIconsEmbed";
```

You will also need to add the provider to the credentials provider list in [`frontend/src/components/integrations/helper.ts`](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform/frontend/src/components/integrations/helper.ts).

frontend/src/components/integrations/helper.ts

```
--8 <
  --"autogpt_platform/frontend/src/components/integrations/helper.ts:CredentialsProviderNames";
```

Finally you will need to add the provider to the `CredentialsType` enum in [`frontend/src/lib/autogpt-server-api/types.ts`](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform/frontend/src/lib/autogpt-server-api/types.ts).

frontend/src/lib/autogpt-server-api/types.ts

```
--8 <
  --"autogpt_platform/frontend/src/lib/autogpt-server-api/types.ts:BlockIOCredentialsSubSchema";
```

**Example: GitHub Integration**

* GitHub blocks with API key + OAuth2 support: [`blocks/github`](https://github.com/Significant-Gravitas/AutoGPT/tree/master/autogpt_platform/backend/backend/blocks/github/)

backend/blocks/github/issues.py

```
class GithubCommentBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        issue_url: str = SchemaField(
            description="URL of the GitHub issue or pull request",
            placeholder="https://github.com/owner/repo/issues/1",
        )
        comment: str = SchemaField(
            description="Comment to post on the issue or pull request",
            placeholder="Enter your comment",
        )

    class Output(BlockSchemaOutput):
        id: int = SchemaField(description="ID of the created comment")
        url: str = SchemaField(description="URL to the comment on GitHub")
        error: str = SchemaField(
            description="Error message if the comment posting failed"
        )

    def __init__(self):
        super().__init__(
            id="a8db4d8d-db1c-4a25-a1b0-416a8c33602b",
            description="This block posts a comment on a specified GitHub issue or pull request.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubCommentBlock.Input,
            output_schema=GithubCommentBlock.Output,
            test_input=[
                {
                    "issue_url": "https://github.com/owner/repo/issues/1",
                    "comment": "This is a test comment.",
                    "credentials": TEST_CREDENTIALS_INPUT,
                },
                {
                    "issue_url": "https://github.com/owner/repo/pull/1",
                    "comment": "This is a test comment.",
                    "credentials": TEST_CREDENTIALS_INPUT,
                },
            ],
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("id", 1337),
                ("url", "https://github.com/owner/repo/issues/1#issuecomment-1337"),
                ("id", 1337),
                (
                    "url",
                    "https://github.com/owner/repo/issues/1#issuecomment-1337",
                ),
            ],
            test_mock={
                "post_comment": lambda *args, **kwargs: (
                    1337,
                    "https://github.com/owner/repo/issues/1#issuecomment-1337",
                )
            },
        )

    @staticmethod
    async def post_comment(
        credentials: GithubCredentials, issue_url: str, body_text: str
    ) -> tuple[int, str]:
        api = get_api(credentials)
        data = {"body": body_text}
        if "pull" in issue_url:
            issue_url = issue_url.replace("pull", "issues")
        comments_url = issue_url + "/comments"
        response = await api.post(comments_url, json=data)
        comment = response.json()
        return comment["id"], comment["html_url"]

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        id, url = await self.post_comment(
            credentials,
            input_data.issue_url,
            input_data.comment,
        )
        yield "id", id
        yield "url", url
```

* GitHub OAuth2 handler: [`integrations/oauth/github.py`](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform/backend/backend/integrations/oauth/github.py)

backend/integrations/oauth/github.py

```
class GitHubOAuthHandler(BaseOAuthHandler):
    """
    Based on the documentation at:
    - [Authorizing OAuth apps - GitHub Docs](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps)
    - [Refreshing user access tokens - GitHub Docs](https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/refreshing-user-access-tokens)

    Notes:
    - By default, token expiration is disabled on GitHub Apps. This means the access
      token doesn't expire and no refresh token is returned by the authorization flow.
    - When token expiration gets enabled, any existing tokens will remain non-expiring.
    - When token expiration gets disabled, token refreshes will return a non-expiring
      access token *with no refresh token*.
    """  # noqa

    PROVIDER_NAME = ProviderName.GITHUB

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.auth_base_url = "https://github.com/login/oauth/authorize"
        self.token_url = "https://github.com/login/oauth/access_token"
        self.revoke_url = "https://api.github.com/applications/{client_id}/token"

    def get_login_url(
        self, scopes: list[str], state: str, code_challenge: Optional[str]
    ) -> str:
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(scopes),
            "state": state,
        }
        return f"{self.auth_base_url}?{urlencode(params)}"

    async def exchange_code_for_tokens(
        self, code: str, scopes: list[str], code_verifier: Optional[str]
    ) -> OAuth2Credentials:
        return await self._request_tokens(
            {"code": code, "redirect_uri": self.redirect_uri}
        )

    async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        if not credentials.access_token:
            raise ValueError("No access token to revoke")

        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        await Requests().delete(
            url=self.revoke_url.format(client_id=self.client_id),
            auth=(self.client_id, self.client_secret),
            headers=headers,
            json={"access_token": credentials.access_token.get_secret_value()},
        )
        return True

    async def _refresh_tokens(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        if not credentials.refresh_token:
            return credentials

        return await self._request_tokens(
            {
                "refresh_token": credentials.refresh_token.get_secret_value(),
                "grant_type": "refresh_token",
            }
        )

    async def _request_tokens(
        self,
        params: dict[str, str],
        current_credentials: Optional[OAuth2Credentials] = None,
    ) -> OAuth2Credentials:
        request_body = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            **params,
        }
        headers = {"Accept": "application/json"}
        response = await Requests().post(
            self.token_url, data=request_body, headers=headers
        )
        token_data: dict = response.json()

        username = await self._request_username(token_data["access_token"])

        now = int(time.time())
        new_credentials = OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            title=current_credentials.title if current_credentials else None,
            username=username,
            access_token=token_data["access_token"],
            # Token refresh responses have an empty `scope` property (see docs),
            # so we have to get the scope from the existing credentials object.
            scopes=(
                token_data.get("scope", "").split(",")
                or (current_credentials.scopes if current_credentials else [])
            ),
            # Refresh token and expiration intervals are only given if token expiration
            # is enabled in the GitHub App's settings.
            refresh_token=token_data.get("refresh_token"),
            access_token_expires_at=(
                now + expires_in
                if (expires_in := token_data.get("expires_in", None))
                else None
            ),
            refresh_token_expires_at=(
                now + expires_in
                if (expires_in := token_data.get("refresh_token_expires_in", None))
                else None
            ),
        )
        if current_credentials:
            new_credentials.id = current_credentials.id
        return new_credentials

    async def _request_username(self, access_token: str) -> str | None:
        url = "https://api.github.com/user"
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {access_token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        response = await Requests().get(url, headers=headers)

        if not response.ok:
            return None

        # Get the login (username)
        resp = response.json()
        return resp.get("login")
```

**Example: Google Integration**

* Google OAuth2 handler: [`integrations/oauth/google.py`](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform/backend/backend/integrations/oauth/google.py)

backend/integrations/oauth/google.py

```
class GoogleOAuthHandler(BaseOAuthHandler):
    """
    Based on the documentation at https://developers.google.com/identity/protocols/oauth2/web-server
    """  # noqa

    PROVIDER_NAME = ProviderName.GOOGLE
    EMAIL_ENDPOINT = "https://www.googleapis.com/oauth2/v2/userinfo"
    DEFAULT_SCOPES = [
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "openid",
    ]
```

You can see that google has defined a `DEFAULT_SCOPES` variable, this is used to set the scopes that are requested no matter what the user asks for.

backend/blocks/google/\_auth.py

```
secrets = Secrets()
GOOGLE_OAUTH_IS_CONFIGURED = bool(
    secrets.google_client_id and secrets.google_client_secret
)
```

You can also see that `GOOGLE_OAUTH_IS_CONFIGURED` is used to disable the blocks that require OAuth if the oauth is not configured. This is in the `__init__` method of each block. This is because there is no api key fallback for google blocks so we need to make sure that the oauth is configured before we allow the user to use the blocks.

#### Webhook-Triggered Blocks <a href="#webhook-triggered-blocks" id="webhook-triggered-blocks"></a>

Webhook-triggered blocks allow your agent to respond to external events in real-time. These blocks are triggered by incoming webhooks from third-party services rather than being executed manually.

Creating and running a webhook-triggered block involves three main components:

* The block itself, which specifies:
* Inputs for the user to select a resource and events to subscribe to
* A `credentials` input with the scopes needed to manage webhooks
* Logic to turn the webhook payload into outputs for the webhook block
* The `WebhooksManager` for the corresponding webhook service provider, which handles:
* (De)registering webhooks with the provider
* Parsing and validating incoming webhook payloads
* The credentials system for the corresponding service provider, which may include an `OAuthHandler`

There is more going on under the hood, e.g. to store and retrieve webhooks and their links to nodes, but to add a webhook-triggered block you shouldn't need to make changes to those parts of the system.

**Creating a Webhook-Triggered Block**

To create a webhook-triggered block, follow these additional steps on top of the basic block creation process:

1. **Define `webhook_config`** in your block's `__init__` method.

<details>

<summary>Example: <code>GitHubPullRequestTriggerBlock</code></summary>

backend/blocks/github/triggers.py

```
webhook_config=BlockWebhookConfig(
    provider=ProviderName.GITHUB,
    webhook_type=GithubWebhookType.REPO,
    resource_format="{repo}",
    event_filter_input="events",
    event_format="pull_request.{event}",
),
```

</details>

<details>

<summary><code>BlockWebhookConfig</code> definition</summary>

backend/data/block.py

```
class BlockManualWebhookConfig(BaseModel):
    """
    Configuration model for webhook-triggered blocks on which
    the user has to manually set up the webhook at the provider.
    """

    provider: ProviderName
    """The service provider that the webhook connects to"""

    webhook_type: str
    """
    Identifier for the webhook type. E.g. GitHub has repo and organization level hooks.

    Only for use in the corresponding `WebhooksManager`.
    """

    event_filter_input: str = ""
    """
    Name of the block's event filter input.
    Leave empty if the corresponding webhook doesn't have distinct event/payload types.
    """

    event_format: str = "{event}"
    """
    Template string for the event(s) that a block instance subscribes to.
    Applied individually to each event selected in the event filter input.

    Example: `"pull_request.{event}"` -> `"pull_request.opened"`
    """


class BlockWebhookConfig(BlockManualWebhookConfig):
    """
    Configuration model for webhook-triggered blocks for which
    the webhook can be automatically set up through the provider's API.
    """

    resource_format: str
    """
    Template string for the resource that a block instance subscribes to.
    Fields will be filled from the block's inputs (except `payload`).

    Example: `f"{repo}/pull_requests"` (note: not how it's actually implemented)

    Only for use in the corresponding `WebhooksManager`.
    """
```

</details>

1. **Define event filter input** in your block's Input schema. This allows the user to select which specific types of events will trigger the block in their agent.

<details>

<summary>Example: <code>GitHubPullRequestTriggerBlock</code></summary>

backend/blocks/github/triggers.py

```
class Input(GitHubTriggerBase.Input):
    class EventsFilter(BaseModel):
        """
        https://docs.github.com/en/webhooks/webhook-events-and-payloads#pull_request
        """

        opened: bool = False
        edited: bool = False
        closed: bool = False
        reopened: bool = False
        synchronize: bool = False
        assigned: bool = False
        unassigned: bool = False
        labeled: bool = False
        unlabeled: bool = False
        converted_to_draft: bool = False
        locked: bool = False
        unlocked: bool = False
        enqueued: bool = False
        dequeued: bool = False
        milestoned: bool = False
        demilestoned: bool = False
        ready_for_review: bool = False
        review_requested: bool = False
        review_request_removed: bool = False
        auto_merge_enabled: bool = False
        auto_merge_disabled: bool = False

    events: EventsFilter = SchemaField(
        title="Events", description="The events to subscribe to"
    )
```

</details>

* The name of the input field (`events` in this case) must match `webhook_config.event_filter_input`.
* The event filter itself must be a Pydantic model with only boolean fields.
* **Include payload field** in your block's Input schema.

<details>

<summary>Example: <code>GitHubTriggerBase</code></summary>

backend/blocks/github/triggers.py

```
payload: dict = SchemaField(hidden=True, default_factory=dict)
```

</details>

1. **Define `credentials` input** in your block's Input schema.
2. Its scopes must be sufficient to manage a user's webhooks through the provider's API
3. See [Blocks with authentication](https://docs.agpt.co/platform/new_blocks/#blocks-with-authentication) for further details
4. **Process webhook payload** and output relevant parts of it in your block's `run` method.

<details>

<summary>Example: <code>GitHubPullRequestTriggerBlock</code></summary>

```
def run(self, input_data: Input, **kwargs) -> BlockOutput:
    yield "payload", input_data.payload
    yield "sender", input_data.payload["sender"]
    yield "event", input_data.payload["action"]
    yield "number", input_data.payload["number"]
    yield "pull_request", input_data.payload["pull_request"]
```

Note that the \`credentials\` parameter can be omitted if the credentials aren't used at block runtime, like in the example.

</details>

**Adding a Webhooks Manager**

To add support for a new webhook provider, you'll need to create a WebhooksManager that implements the `BaseWebhooksManager` interface:

backend/integrations/webhooks/\_base.py

```
PROVIDER_NAME: ClassVar[ProviderName]

@abstractmethod
async def _register_webhook(
    self,
    credentials: Credentials,
    webhook_type: WT,
    resource: str,
    events: list[str],
    ingress_url: str,
    secret: str,
) -> tuple[str, dict]:
    """
    Registers a new webhook with the provider.

    Params:
        credentials: The credentials with which to create the webhook
        webhook_type: The provider-specific webhook type to create
        resource: The resource to receive events for
        events: The events to subscribe to
        ingress_url: The ingress URL for webhook payloads
        secret: Secret used to verify webhook payloads

    Returns:
        str: Webhook ID assigned by the provider
        config: Provider-specific configuration for the webhook
    """
    ...

@classmethod
@abstractmethod
async def validate_payload(
    cls,
    webhook: integrations.Webhook,
    request: Request,
    credentials: Credentials | None,
) -> tuple[dict, str]:
    """
    Validates an incoming webhook request and returns its payload and type.

    Params:
        webhook: Object representing the configured webhook and its properties in our system.
        request: Incoming FastAPI `Request`

    Returns:
        dict: The validated payload
        str: The event type associated with the payload
    """

@abstractmethod
async def _deregister_webhook(
    self, webhook: integrations.Webhook, credentials: Credentials
) -> None: ...

async def trigger_ping(
    self, webhook: integrations.Webhook, credentials: Credentials | None
) -> None:
    """
    Triggers a ping to the given webhook.

    Raises:
        NotImplementedError: if the provider doesn't support pinging
    """
```

And add a reference to your `WebhooksManager` class in `load_webhook_managers`:

backend/integrations/webhooks/\_\_init\_\_.py

```
@cached(ttl_seconds=3600)
def load_webhook_managers() -> dict["ProviderName", type["BaseWebhooksManager"]]:
    webhook_managers = {}

    from .compass import CompassWebhookManager
    from .github import GithubWebhooksManager
    from .slant3d import Slant3DWebhooksManager

    webhook_managers.update(
        {
            handler.PROVIDER_NAME: handler
            for handler in [
                CompassWebhookManager,
                GithubWebhooksManager,
                Slant3DWebhooksManager,
            ]
        }
    )
    return webhook_managers
```

**Example: GitHub Webhook Integration**

<details>

<summary>GitHub Webhook triggers: <a href="https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform/backend/backend/blocks/github/triggers.py"><code>blocks/github/triggers.py</code></a></summary>

backend/blocks/github/triggers.py

```
class GitHubTriggerBase:
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo: str = SchemaField(
            description=(
                "Repository to subscribe to.\n\n"
                "**Note:** Make sure your GitHub credentials have permissions "
                "to create webhooks on this repo."
            ),
            placeholder="{owner}/{repo}",
        )
        payload: dict = SchemaField(hidden=True, default_factory=dict)

    class Output(BlockSchemaOutput):
        payload: dict = SchemaField(
            description="The complete webhook payload that was received from GitHub. "
            "Includes information about the affected resource (e.g. pull request), "
            "the event, and the user who triggered the event."
        )
        triggered_by_user: dict = SchemaField(
            description="Object representing the GitHub user who triggered the event"
        )
        error: str = SchemaField(
            description="Error message if the payload could not be processed"
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        yield "payload", input_data.payload
        yield "triggered_by_user", input_data.payload["sender"]


class GithubPullRequestTriggerBlock(GitHubTriggerBase, Block):
    EXAMPLE_PAYLOAD_FILE = (
        Path(__file__).parent / "example_payloads" / "pull_request.synchronize.json"
    )

    class Input(GitHubTriggerBase.Input):
        class EventsFilter(BaseModel):
            """
            https://docs.github.com/en/webhooks/webhook-events-and-payloads#pull_request
            """

            opened: bool = False
            edited: bool = False
            closed: bool = False
            reopened: bool = False
            synchronize: bool = False
            assigned: bool = False
            unassigned: bool = False
            labeled: bool = False
            unlabeled: bool = False
            converted_to_draft: bool = False
            locked: bool = False
            unlocked: bool = False
            enqueued: bool = False
            dequeued: bool = False
            milestoned: bool = False
            demilestoned: bool = False
            ready_for_review: bool = False
            review_requested: bool = False
            review_request_removed: bool = False
            auto_merge_enabled: bool = False
            auto_merge_disabled: bool = False

        events: EventsFilter = SchemaField(
            title="Events", description="The events to subscribe to"
        )

    class Output(GitHubTriggerBase.Output):
        event: str = SchemaField(
            description="The PR event that triggered the webhook (e.g. 'opened')"
        )
        number: int = SchemaField(description="The number of the affected pull request")
        pull_request: dict = SchemaField(
            description="Object representing the affected pull request"
        )
        pull_request_url: str = SchemaField(
            description="The URL of the affected pull request"
        )

    def __init__(self):
        from backend.integrations.webhooks.github import GithubWebhookType

        example_payload = json.loads(
            self.EXAMPLE_PAYLOAD_FILE.read_text(encoding="utf-8")
        )

        super().__init__(
            id="6c60ec01-8128-419e-988f-96a063ee2fea",
            description="This block triggers on pull request events and outputs the event type and payload.",
            categories={BlockCategory.DEVELOPER_TOOLS, BlockCategory.INPUT},
            input_schema=GithubPullRequestTriggerBlock.Input,
            output_schema=GithubPullRequestTriggerBlock.Output,
            webhook_config=BlockWebhookConfig(
                provider=ProviderName.GITHUB,
                webhook_type=GithubWebhookType.REPO,
                resource_format="{repo}",
                event_filter_input="events",
                event_format="pull_request.{event}",
            ),
            test_input={
                "repo": "Significant-Gravitas/AutoGPT",
                "events": {"opened": True, "synchronize": True},
                "credentials": TEST_CREDENTIALS_INPUT,
                "payload": example_payload,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("payload", example_payload),
                ("triggered_by_user", example_payload["sender"]),
                ("event", example_payload["action"]),
                ("number", example_payload["number"]),
                ("pull_request", example_payload["pull_request"]),
                ("pull_request_url", example_payload["pull_request"]["html_url"]),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:  # type: ignore
        async for name, value in super().run(input_data, **kwargs):
            yield name, value
        yield "event", input_data.payload["action"]
        yield "number", input_data.payload["number"]
        yield "pull_request", input_data.payload["pull_request"]
        yield "pull_request_url", input_data.payload["pull_request"]["html_url"]
```

</details>

<details>

<summary>GitHub Webhooks Manager: <a href="https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform/backend/backend/integrations/webhooks/github.py"><code>integrations/webhooks/github.py</code></a></summary>

backend/integrations/webhooks/github.py

```
class GithubWebhookType(StrEnum):
    REPO = "repo"


class GithubWebhooksManager(BaseWebhooksManager):
    PROVIDER_NAME = ProviderName.GITHUB

    WebhookType = GithubWebhookType

    GITHUB_API_URL = "https://api.github.com"
    GITHUB_API_DEFAULT_HEADERS = {"Accept": "application/vnd.github.v3+json"}

    @classmethod
    async def validate_payload(
        cls,
        webhook: integrations.Webhook,
        request: Request,
        credentials: Credentials | None,
    ) -> tuple[dict, str]:
        if not (event_type := request.headers.get("X-GitHub-Event")):
            raise HTTPException(
                status_code=400, detail="X-GitHub-Event header is missing!"
            )

        if not (signature_header := request.headers.get("X-Hub-Signature-256")):
            raise HTTPException(
                status_code=403, detail="X-Hub-Signature-256 header is missing!"
            )

        payload_body = await request.body()
        hash_object = hmac.new(
            webhook.secret.encode("utf-8"), msg=payload_body, digestmod=hashlib.sha256
        )
        expected_signature = "sha256=" + hash_object.hexdigest()

        if not hmac.compare_digest(expected_signature, signature_header):
            raise HTTPException(
                status_code=403, detail="Request signatures didn't match!"
            )

        payload = await request.json()
        if action := payload.get("action"):
            event_type += f".{action}"

        return payload, event_type

    async def trigger_ping(
        self, webhook: integrations.Webhook, credentials: Credentials | None
    ) -> None:
        if not credentials:
            raise ValueError("Credentials are required but were not passed")

        headers = {
            **self.GITHUB_API_DEFAULT_HEADERS,
            "Authorization": credentials.auth_header(),
        }

        repo, github_hook_id = webhook.resource, webhook.provider_webhook_id
        ping_url = f"{self.GITHUB_API_URL}/repos/{repo}/hooks/{github_hook_id}/pings"

        response = await Requests().post(ping_url, headers=headers)

        if response.status != 204:
            error_msg = extract_github_error_msg(response)
            raise ValueError(f"Failed to ping GitHub webhook: {error_msg}")

    async def _register_webhook(
        self,
        credentials: Credentials,
        webhook_type: GithubWebhookType,
        resource: str,
        events: list[str],
        ingress_url: str,
        secret: str,
    ) -> tuple[str, dict]:
        if webhook_type == self.WebhookType.REPO and resource.count("/") > 1:
            raise ValueError("Invalid repo format: expected 'owner/repo'")

        # Extract main event, e.g. `pull_request.opened` -> `pull_request`
        github_events = list({event.split(".")[0] for event in events})

        headers = {
            **self.GITHUB_API_DEFAULT_HEADERS,
            "Authorization": credentials.auth_header(),
        }
        webhook_data = {
            "name": "web",
            "active": True,
            "events": github_events,
            "config": {
                "url": ingress_url,
                "content_type": "json",
                "insecure_ssl": "0",
                "secret": secret,
            },
        }

        response = await Requests().post(
            f"{self.GITHUB_API_URL}/repos/{resource}/hooks",
            headers=headers,
            json=webhook_data,
        )

        if response.status != 201:
            error_msg = extract_github_error_msg(response)
            if "not found" in error_msg.lower():
                error_msg = (
                    f"{error_msg} "
                    "(Make sure the GitHub account or API key has 'repo' or "
                    f"webhook create permissions to '{resource}')"
                )
            raise ValueError(f"Failed to create GitHub webhook: {error_msg}")

        resp = response.json()
        webhook_id = resp["id"]
        config = resp["config"]

        return str(webhook_id), config

    async def _deregister_webhook(
        self, webhook: integrations.Webhook, credentials: Credentials
    ) -> None:
        webhook_type = self.WebhookType(webhook.webhook_type)
        if webhook.credentials_id != credentials.id:
            raise ValueError(
                f"Webhook #{webhook.id} does not belong to credentials {credentials.id}"
            )

        headers = {
            **self.GITHUB_API_DEFAULT_HEADERS,
            "Authorization": credentials.auth_header(),
        }

        if webhook_type == self.WebhookType.REPO:
            repo = webhook.resource
            delete_url = f"{self.GITHUB_API_URL}/repos/{repo}/hooks/{webhook.provider_webhook_id}"  # noqa
        else:
            raise NotImplementedError(
                f"Unsupported webhook type '{webhook.webhook_type}'"
            )

        response = await Requests().delete(delete_url, headers=headers)

        if response.status not in [204, 404]:
            # 204 means successful deletion, 404 means the webhook was already deleted
            error_msg = extract_github_error_msg(response)
            raise ValueError(f"Failed to delete GitHub webhook: {error_msg}")

        # If we reach here, the webhook was successfully deleted or didn't exist
```

</details>

### Key Points to Remember <a href="#key-points-to-remember" id="key-points-to-remember"></a>

* **Unique ID**: Give your block a unique ID in the **init** method.
* **Input and Output Schemas**: Define clear input and output schemas.
* **Error Handling**: Implement error handling in the `run` method.
* **Output Results**: Use `yield` to output results in the `run` method.
* **Testing**: Provide test input and output in the **init** method for automatic testing.

### Understanding the Testing Process <a href="#understanding-the-testing-process" id="understanding-the-testing-process"></a>

The testing of blocks is handled by `test_block.py`, which does the following:

1. It calls the block with the provided `test_input`. If the block has a `credentials` field, `test_credentials` is passed in as well.
2. If a `test_mock` is provided, it temporarily replaces the specified methods with the mock functions.
3. It then asserts that the output matches the `test_output`.

For the WikipediaSummaryBlock:

* The test will call the block with the topic "Artificial Intelligence".
* Instead of making a real API call, it will use the mock function, which returns `{"extract": "summary content"}`.
* It will then check if the output key is "summary" and its value is a string.

This approach allows us to test the block's logic comprehensively without relying on external services, while also accommodating non-deterministic outputs.

### Security Best Practices for SSRF Prevention <a href="#security-best-practices-for-ssrf-prevention" id="security-best-practices-for-ssrf-prevention"></a>

When creating blocks that handle external URL inputs or make network requests, it's crucial to use the platform's built-in SSRF protection mechanisms. The `backend.util.request` module provides a secure `Requests` wrapper class that should be used for all HTTP requests.

#### Using the Secure Requests Wrapper <a href="#using-the-secure-requests-wrapper" id="using-the-secure-requests-wrapper"></a>

```
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
2. Blocks requests to private IP ranges (RFC 1918)
3. Validates URL format and protocol
4. Resolves DNS and checks IP addresses
5. Supports whitelisting trusted origins
6. **Secure Defaults**:
7. Disables redirects by default
8. Raises exceptions for non-200 status codes
9. Supports custom headers and validators
10. **Protected IP Ranges**: The wrapper denies requests to these networks:

backend/util/request.py

```
# IPv4 Ranges
ipaddress.ip_network("0.0.0.0/8"),  # "This" Network
ipaddress.ip_network("10.0.0.0/8"),  # Private-Use
ipaddress.ip_network("127.0.0.0/8"),  # Loopback
ipaddress.ip_network("169.254.0.0/16"),  # Link Local
ipaddress.ip_network("172.16.0.0/12"),  # Private-Use
ipaddress.ip_network("192.168.0.0/16"),  # Private-Use
ipaddress.ip_network("224.0.0.0/4"),  # Multicast
ipaddress.ip_network("240.0.0.0/4"),  # Reserved for Future Use
# IPv6 Ranges
ipaddress.ip_network("::1/128"),  # Loopback
ipaddress.ip_network("fc00::/7"),  # Unique local addresses (ULA)
ipaddress.ip_network("fe80::/10"),  # Link-local
ipaddress.ip_network("ff00::/8"),  # Multicast
```

#### Custom Request Configuration <a href="#custom-request-configuration" id="custom-request-configuration"></a>

If you need to customize the request behavior:

```
from backend.util.request import Requests

# Create a custom requests instance with specific trusted origins
custom_requests = Requests(
    trusted_origins=["api.trusted-service.com"],
    raise_for_status=True,
    extra_headers={"User-Agent": "MyBlock/1.0"}
)
```

#### Error Handling <a href="#error-handling" id="error-handling"></a>

All blocks should have an error output that catches all reasonable errors that a user can handle, wrap them in a ValueError, and re-raise. Don't catch things the system admin would need to fix like being out of money or unreachable addresses.

#### Data Models <a href="#data-models" id="data-models"></a>

Use pydantic base models over dict and typeddict where possible. Avoid untyped models for block inputs and outputs as much as possible

#### File Input <a href="#file-input" id="file-input"></a>

You can use MediaFileType to handle the importing and exporting of files out of the system. Explore how its used through the system before using it in a block schema.

### Tips for Effective Block Testing <a href="#tips-for-effective-block-testing" id="tips-for-effective-block-testing"></a>

1. **Provide realistic test\_input**: Ensure your test input covers typical use cases.
2. **Define appropriate test\_output**:
3. For deterministic outputs, use specific expected values.
4. For non-deterministic outputs or when only the type matters, use Python types (e.g., `str`, `int`, `dict`).
5. You can mix specific values and types, e.g., `("key1", str), ("key2", 42)`.
6. **Use test\_mock for network calls**: This prevents tests from failing due to network issues or API changes.
7. **Consider omitting test\_mock for blocks without external dependencies**: If your block doesn't make network calls or use external resources, you might not need a mock.
8. **Consider edge cases**: Include tests for potential error conditions in your `run` method.
9. **Update tests when changing block behavior**: If you modify your block, ensure the tests are updated accordingly.

By following these steps, you can create new blocks that extend the functionality of the AutoGPT Agent Server.
