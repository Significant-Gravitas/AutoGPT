# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Block Development with SDK

The AutoGPT Platform now includes a comprehensive SDK that dramatically simplifies block creation. Blocks can be fully self-contained with zero external configuration required.

### Quick Start - Creating a New Block

```python
from backend.sdk import *

@provider("my-service")  # Auto-registers new provider
@cost_config(
    BlockCost(cost_amount=5, cost_type=BlockCostType.RUN),
    BlockCost(cost_amount=1, cost_type=BlockCostType.BYTE)
)
@default_credentials(
    APIKeyCredentials(
        id="my-service-default",
        provider="my-service",
        api_key=SecretStr("default-api-key"),
        title="My Service Default API Key"
    )
)
class MyServiceBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = CredentialsField(
            provider="my-service",
            supported_credential_types={"api_key"}
        )
        text: String = SchemaField(description="Input text")
        
    class Output(BlockSchema):
        result: String = SchemaField(description="Output result")
        error: String = SchemaField(description="Error message", default="")
    
    def __init__(self):
        super().__init__(
            id="my-service-block-12345678-1234-1234-1234-123456789012",
            description="Process text using My Service",
            categories={BlockCategory.TEXT},
            input_schema=MyServiceBlock.Input,
            output_schema=MyServiceBlock.Output,
        )
    
    def run(self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs) -> BlockOutput:
        try:
            api_key = credentials.api_key.get_secret_value()
            # Process with API
            yield "result", f"Processed: {input_data.text}"
        except Exception as e:
            yield "error", str(e)
```

### Key Features

1. **Single Import**: `from backend.sdk import *` provides everything needed
2. **Auto-Registration**: No manual configuration files to update
3. **Dynamic Providers**: Any string works as a provider name
4. **Self-Contained**: All configuration via decorators

### Available Decorators

- `@provider("name")` - Register new provider
- `@cost_config(...)` - Set block execution costs
- `@default_credentials(...)` - Provide default API credentials
- `@webhook_config("provider", ManagerClass)` - Register webhook manager
- `@oauth_config("provider", HandlerClass)` - Register OAuth handler

### Creating Blocks with Webhooks

```python
from backend.sdk import *

# First, create webhook manager
class MyWebhookManager(BaseWebhooksManager):
    PROVIDER_NAME = "my-service"
    
    class WebhookType(str, Enum):
        DATA_UPDATE = "data_update"
    
    async def validate_payload(self, webhook, request) -> tuple[dict, str]:
        payload = await request.json()
        event_type = request.headers.get("X-MyService-Event", "unknown")
        return payload, event_type
    
    async def _register_webhook(self, webhook, credentials) -> tuple[str, dict]:
        # Register with external service
        return "webhook-id", {"status": "registered"}
    
    async def _deregister_webhook(self, webhook, credentials) -> None:
        # Deregister from external service
        pass

# Then create webhook block
@provider("my-service")
@webhook_config("my-service", MyWebhookManager)
class MyWebhookBlock(Block):
    class Input(BlockSchema):
        events: BaseModel = SchemaField(
            description="Events to listen for",
            default={"data_update": True}
        )
        payload: Dict = SchemaField(
            description="Webhook payload",
            default={},
            hidden=True
        )
        
    class Output(BlockSchema):
        event_type: String = SchemaField(description="Event type")
        event_data: Dict = SchemaField(description="Event data")
    
    def __init__(self):
        super().__init__(
            id="my-webhook-block-12345678-1234-1234-1234-123456789012",
            description="Listen for My Service webhooks",
            categories={BlockCategory.INPUT},
            input_schema=MyWebhookBlock.Input,
            output_schema=MyWebhookBlock.Output,
            block_type=BlockType.WEBHOOK,
            webhook_config=BlockWebhookConfig(
                provider="my-service",
                webhook_type="data_update",
                event_filter_input="events",
            ),
        )
    
    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        payload = input_data.payload
        yield "event_type", payload.get("type", "unknown")
        yield "event_data", payload
```

### Creating Blocks with OAuth

```python
from backend.sdk import *

# First, create OAuth handler
class MyServiceOAuthHandler(BaseOAuthHandler):
    PROVIDER_NAME = "my-service"
    DEFAULT_SCOPES = ["read", "write"]
    
    def get_login_url(self, scopes: list[str], state: str, code_challenge: Optional[str]) -> str:
        # Build OAuth authorization URL
        return f"https://my-service.com/oauth/authorize?..."
    
    def exchange_code_for_tokens(self, code: str, scopes: list[str], code_verifier: Optional[str]) -> OAuth2Credentials:
        # Exchange authorization code for tokens
        return OAuth2Credentials(
            provider="my-service",
            access_token=SecretStr("access-token"),
            refresh_token=SecretStr("refresh-token"),
            scopes=scopes,
            access_token_expires_at=int(time.time() + 3600)
        )

# Then create OAuth-enabled block
@provider("my-service")
@oauth_config("my-service", MyServiceOAuthHandler)
class MyOAuthBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = CredentialsField(
            provider="my-service",
            supported_credential_types={"oauth2"},
            required_scopes={"read", "write"}
        )
        action: String = SchemaField(description="Action to perform")
        
    class Output(BlockSchema):
        result: Dict = SchemaField(description="API response")
        error: String = SchemaField(description="Error message", default="")
    
    def __init__(self):
        super().__init__(
            id="my-oauth-block-12345678-1234-1234-1234-123456789012",
            description="Interact with My Service using OAuth",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=MyOAuthBlock.Input,
            output_schema=MyOAuthBlock.Output,
        )
    
    def run(self, input_data: Input, *, credentials: OAuth2Credentials, **kwargs) -> BlockOutput:
        try:
            headers = {"Authorization": f"Bearer {credentials.access_token.get_secret_value()}"}
            # Make API call with OAuth token
            yield "result", {"status": "success", "action": input_data.action}
        except Exception as e:
            yield "error", str(e)
```

### SDK Components Available

The SDK provides 68+ components via `from backend.sdk import *`:

**Core Block Components:**
- `Block`, `BlockSchema`, `BlockOutput`, `BlockCategory`, `BlockType`
- `SchemaField`, `CredentialsField`, `CredentialsMetaInput`

**Credential Types:**
- `APIKeyCredentials`, `OAuth2Credentials`, `UserPasswordCredentials`

**Cost System:**
- `BlockCost`, `BlockCostType`, `NodeExecutionStats`

**Type Aliases:**
- `String`, `Integer`, `Float`, `Boolean` (cleaner than str, int, etc.)

**Common Types:**
- `List`, `Dict`, `Optional`, `Any`, `Union`, `BaseModel`, `SecretStr`

**Utilities:**
- `json`, `logging`, `store_media_file`, `MediaFileType`

### Best Practices

1. **Use UUID for Block ID**: Generate a unique UUID for each block
2. **Handle Errors**: Always include error handling in the run method
3. **Yield All Outputs**: Ensure all output schema fields are yielded
4. **Test Your Block**: Include test_input and test_output in __init__
5. **Document Well**: Provide clear descriptions for the block and all fields

### No Manual Configuration Needed

With the SDK, you never need to manually update these files:
- ❌ `backend/data/block_cost_config.py`
- ❌ `backend/integrations/credentials_store.py`
- ❌ `backend/integrations/providers.py`
- ❌ `backend/integrations/oauth/__init__.py`
- ❌ `backend/integrations/webhooks/__init__.py`

Everything is handled automatically by the decorators!

## Project Architecture

The AutoGPT Platform is a microservice-based system for creating and running AI-powered agent workflows. It consists of three main components:

### Core Components

- **Backend** (`backend/`): Python FastAPI microservices with Redis, RabbitMQ, and PostgreSQL
- **Frontend** (`frontend/`): Next.js 14 application with TypeScript and Radix UI components  
- **Shared Libraries** (`autogpt_libs/`): Common Python utilities for auth, logging, rate limiting

### Service Architecture

The backend runs multiple services that communicate via Redis and RabbitMQ:

- **REST API Server** (port 8006-8007): Main HTTP API endpoints
- **WebSocket Server** (port 8001): Real-time communication for frontend
- **Executor** (port 8002): Handles workflow execution with block-based architecture
- **Scheduler** (port 8003): Manages scheduled agent runs
- **Database Manager**: Handles migrations and database connections
- **Notification Manager**: Email notifications and user alerts

### Data Model

- **AgentGraph**: Core workflow definition with nodes and links
- **AgentGraphExecution**: Runtime execution instances with status tracking
- **User**: Authentication via Supabase with credit system and integrations
- **Block**: Individual workflow components (400+ integrations supported)
- **LibraryAgent**: Reusable agent templates
- **StoreListing**: Marketplace for sharing agents

## Development Commands

### Backend Development
```bash
cd backend
poetry install
poetry run app          # All services
poetry run rest         # REST API only
poetry run ws           # WebSocket only
poetry run executor     # Executor only
poetry run scheduler    # Scheduler only
poetry run format       # Black + isort formatting
poetry run lint         # Ruff linting
poetry run test         # Run tests with Docker PostgreSQL
```

### Frontend Development
```bash
cd frontend
npm install
npm run dev             # Development server (port 3000)
npm run build           # Production build
npm run lint            # ESLint + Prettier
npm run format          # Prettier only
npm run type-check      # TypeScript checking
npm run test            # Playwright E2E tests
npm run test-ui         # Playwright UI mode
npm run storybook       # Component development (port 6006)
```

### Docker Operations
```bash
docker compose up -d              # Start all backend services
docker compose stop               # Stop services
docker compose down               # Stop and remove containers
docker compose logs -f <service>  # View service logs
docker compose build <service>    # Rebuild specific service
```

### Database Management
```bash
cd backend
poetry run prisma migrate dev     # Apply migrations
poetry run prisma generate        # Generate Prisma client
poetry run prisma db push         # Push schema changes
```

## Code Architecture Patterns

### Block System
The core execution model uses a block-based architecture where each block represents an atomic operation:

- Blocks inherit from `backend.blocks.block.Block`
- Input/Output schemas defined using Pydantic models
- Blocks are auto-discovered and registered at runtime
- Each block has a unique UUID and category classification

### Data Layer
- **Prisma ORM** for PostgreSQL with Python async client
- **Repository pattern** in `backend/data/` modules
- **Pydantic models** for API serialization in `backend/data/model.py`
- **Database connection pooling** via `backend/data/db.py`

### API Architecture
- **FastAPI** with automatic OpenAPI generation
- **WebSocket support** for real-time execution updates
- **Supabase integration** for authentication and row-level security
- **Middleware** for auth, CORS, rate limiting in `autogpt_libs/`

### Frontend Architecture
- **Next.js App Router** with TypeScript
- **React Flow** for visual workflow builder (`@xyflow/react`)
- **Zustand/React Context** for state management
- **Radix UI** components with Tailwind CSS styling
- **Supabase client** for auth and real-time subscriptions

## Environment Setup

### Required Environment Variables

**Backend (.env)**:
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_HOST` - Redis server for caching/sessions
- `RABBITMQ_HOST` - RabbitMQ for async messaging
- `SUPABASE_URL` + `SUPABASE_JWT_SECRET` - Authentication
- `ENABLE_AUTH=true` - Enable Supabase authentication

**Frontend (.env.local)**:
- `NEXT_PUBLIC_AGPT_SERVER_URL` - Backend REST API URL
- `NEXT_PUBLIC_AGPT_WS_SERVER_URL` - Backend WebSocket URL
- `NEXT_PUBLIC_SUPABASE_URL` + `NEXT_PUBLIC_SUPABASE_ANON_KEY` - Auth

### Integration Setup
The platform supports 400+ integrations requiring various API keys:
- **AI Providers**: OpenAI, Anthropic, Groq, Replicate
- **OAuth Providers**: GitHub, Google, Linear, Twitter, Todoist
- **Business Tools**: Stripe, HubSpot, Discord, Reddit

## Testing Strategy

### Backend Testing
- **pytest** with async support for unit/integration tests
- **Docker PostgreSQL** instance for database tests
- **Faker** for test data generation
- Run tests: `poetry run test`

### Frontend Testing  
- **Playwright** for end-to-end testing
- **Storybook** for component testing and documentation
- **TypeScript** strict mode for compile-time safety
- Run tests: `npm run test` or `npm run test-ui`

## Development Workflow

1. **Start backend services**: `docker compose up -d`
2. **Start frontend**: `cd frontend && npm run dev`
3. **Access application**: http://localhost:3000
4. **View Storybook**: http://localhost:6006
5. **Monitor logs**: `docker compose logs -f <service>`

### Code Quality
- **Backend**: Use `poetry run format` then `poetry run lint` before commits
- **Frontend**: Use `npm run format` then `npm run lint` before commits
- **Type checking**: Run `npm run type-check` for frontend TypeScript validation

### Database Changes
1. Edit `schema.prisma` file
2. Run `poetry run prisma migrate dev --name <migration_name>`
3. Commit both schema and migration files

## Performance Considerations

- **Executor scaling**: Use `docker compose up -d --scale executor=3` for high load
- **Redis caching**: Implemented for user sessions and API responses
- **Database indexing**: Key indexes on user_id, execution_id, created_at fields
- **Frontend optimization**: Next.js build includes automatic code splitting