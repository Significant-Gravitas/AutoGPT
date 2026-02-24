# OAuth Integration Flow Documentation

## Overview

The AutoGPT platform implements OAuth 2.0 in two distinct contexts:

1. **User Authentication (SSO)**: Handled by Supabase for platform login
2. **API Integration Credentials**: Custom OAuth implementation for third-party service access

This document focuses on the **API Integration OAuth flow** used for connecting to external services. For the list of supported providers, see `/backend/backend/integrations/providers.py`. For user authentication documentation, see the Supabase auth implementation.

## Trust Boundaries

### 1. Frontend Trust Boundary
- **Location**: Browser/Client-side application
- **Components**: 
  - `CredentialsInput` component (`/frontend/src/components/integrations/credentials-input.tsx`)
  - OAuth callback route (`/frontend/src/app/(platform)/auth/integrations/oauth_callback/route.ts`)
- **Trust Level**: Untrusted - user-controlled environment
- **Security Measures**:
  - CSRF protection via state tokens
  - Popup-based flow to prevent URL exposure
  - Message validation for cross-window communication

### 2. Backend API Trust Boundary
- **Location**: Server-side FastAPI application
- **Components**:
  - Integration router (`/backend/backend/api/features/integrations/router.py`)
  - OAuth handlers (`/backend/backend/integrations/oauth/`)
  - Credentials store (`/backend/backend/integrations/credentials_store.py`)
- **Trust Level**: Trusted - server-controlled environment
- **Security Measures**:
  - JWT-based authentication
  - Encrypted credential storage
  - Token refresh handling
  - Scope validation

### 3. External Provider Trust Boundary
- **Location**: Third-party OAuth providers
- **Components**: Provider authorization endpoints
- **Trust Level**: Semi-trusted - external services
- **Security Measures**:
  - HTTPS-only communication
  - Provider-specific security features
  - Token revocation support

## Component Architecture

### Frontend Components

#### 1. CredentialsInput Component
- **Purpose**: UI component for credential selection and OAuth initiation
- **Key Functions**:
  - Displays available credentials
  - Initiates OAuth flow via popup window
  - Handles OAuth callback messages
  - Manages credential selection state

#### 2. OAuth Callback Route
- **Path**: `/auth/integrations/oauth_callback`
- **Purpose**: Receives OAuth authorization codes from providers
- **Flow**:
  1. Receives `code` and `state` parameters from provider
  2. Posts message to parent window with results
  3. Auto-closes popup window

### Backend Components

#### 1. Integration Router
- **Base Path**: `/api/integrations`
- **Key Endpoints**:
  - `GET /{provider}/login` - Initiates OAuth flow
  - `POST /{provider}/callback` - Exchanges auth code for tokens
  - `GET /credentials` - Lists user credentials
  - `DELETE /{provider}/credentials/{id}` - Revokes credentials

#### 2. OAuth Base Handler
- **Purpose**: Abstract base class for provider-specific OAuth implementations
- **Key Methods**:
  - `get_login_url()` - Constructs provider authorization URL
  - `exchange_code_for_tokens()` - Exchanges auth code for access tokens
  - `refresh_tokens()` - Refreshes expired access tokens
  - `revoke_tokens()` - Revokes tokens at provider

#### 3. Credentials Store
- **Purpose**: Manages credential persistence and state
- **Key Features**:
  - Redis-backed mutex for concurrent access control
  - OAuth state token generation and validation
  - PKCE support with code challenge generation
  - Default system credentials injection

## OAuth Flow Sequence

### 1. Flow Initiation
```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant Backend
    participant Redis
    participant Provider

    User->>Frontend: Click "Sign in with Provider"
    Frontend->>Backend: GET /api/integrations/{provider}/login
    Backend->>Redis: Store state token + code verifier
    Backend->>Frontend: Return login URL + state token
    Frontend->>Frontend: Open popup window
    Frontend->>Provider: Redirect to authorization URL
```

### 2. Authorization
```mermaid
sequenceDiagram
    participant User
    participant Provider
    participant Callback
    participant Frontend
    participant Backend

    User->>Provider: Authorize application
    Provider->>Callback: Redirect with code + state
    Callback->>Frontend: PostMessage with code + state
    Frontend->>Backend: POST /api/integrations/{provider}/callback
    Backend->>Provider: Exchange code for tokens
    Provider->>Backend: Return access + refresh tokens
    Backend->>Backend: Store credentials
    Backend->>Frontend: Return credential metadata
```

### 3. Token Refresh
```mermaid
sequenceDiagram
    participant Application
    participant Backend
    participant Provider

    Application->>Backend: Request with credential ID
    Backend->>Backend: Check token expiry
    Backend->>Provider: POST refresh token
    Provider->>Backend: Return new tokens
    Backend->>Backend: Update stored credentials
    Backend->>Application: Return valid access token
```

## System Architecture Diagram

```mermaid
graph TB
    subgraph "OAuth Use Cases"
        subgraph "User SSO Login"
            LP[Login Page]
            SB[Supabase Auth]
            GO[Google OAuth SSO]
            SC[Session Cookies]
        end
        
        subgraph "API Integration OAuth"
            UI[CredentialsInput Component]
            CB[OAuth Callback Route]
            PW[Popup Window]
        end
    end
    
    subgraph "Backend API (Trusted)"
        subgraph "Auth Management"
            SA[Supabase Client]
            UM[User Management]
        end
        
        subgraph "Integration Management"
            IR[Integration Router]
            OH[OAuth Handlers]
            CS[Credentials Store]
            CM[Credentials Manager]
        end
    end
    
    subgraph "Storage"
        RD[(Redis)]
        PG[(PostgreSQL)]
        SDB[(Supabase DB)]
    end
    
    subgraph "External Providers"
        GH[GitHub OAuth]
        GL[Google APIs OAuth]
        NT[Notion OAuth]
        OT[...Other Providers]
    end
    
    %% User Login Flow
    LP -->|Login with Google| SB
    SB -->|OAuth Request| GO
    GO -->|User Auth| SB
    SB -->|Session| SC
    SB -->|User Data| SDB
    
    %% API Integration Flow
    UI -->|1. Initiate OAuth| IR
    IR -->|2. Generate State| RD
    IR -->|3. Return Auth URL| UI
    UI -->|4. Open Popup| PW
    PW -->|5. Redirect| GH
    GH -->|6. Auth Code| CB
    CB -->|7. PostMessage| UI
    UI -->|8. Send Code| IR
    IR -->|9. Exchange Code| OH
    OH -->|10. Get Tokens| GH
    OH -->|11. Store Creds| CS
    CS -->|12. Save| PG
    
    OH -.->|Token Refresh| GL
    OH -.->|Token Refresh| NT
    OH -.->|Token Refresh| OT
```

## Data Flow Diagram

```mermaid
graph LR
    subgraph "Data Types"
        ST[State Token]
        CV[Code Verifier]
        CC[Code Challenge]
        AC[Auth Code]
        AT[Access Token]
        RT[Refresh Token]
    end
    
    subgraph "Frontend Flow"
        U1[User Initiates]
        U2[Receives State]
        U3[Opens Popup]
        U4[Receives Code]
        U5[Sends to Backend]
    end
    
    subgraph "Backend Flow"
        B1[Generate State]
        B2[Store in Redis]
        B3[Validate State]
        B4[Exchange Code]
        B5[Store Credentials]
    end
    
    U1 --> B1
    B1 --> ST
    B1 --> CV
    CV --> CC
    B2 --> U2
    U3 --> AC
    AC --> U4
    U5 --> B3
    B3 --> B4
    B4 --> AT
    B4 --> RT
    AT --> B5
    RT --> B5
```

## Security Architecture

```mermaid
graph TB
    subgraph "Security Layers"
        subgraph "Transport Security"
            HTTPS[HTTPS Only]
            CSP[Content Security Policy]
        end
        
        subgraph "Authentication"
            JWT[JWT Tokens]
            STATE[CSRF State Tokens]
            PKCE[PKCE Challenge]
        end
        
        subgraph "Storage Security"
            ENC[Encrypted Credentials]
            SEC[SecretStr Type]
            MUTEX[Redis Mutex Locks]
        end
        
        subgraph "Access Control"
            USER[User Scoped]
            SCOPE[OAuth Scopes]
            EXPIRE[Token Expiration]
        end
    end
    
    HTTPS --> JWT
    JWT --> USER
    STATE --> PKCE
    PKCE --> ENC
    ENC --> SEC
    SEC --> MUTEX
    USER --> SCOPE
    SCOPE --> EXPIRE
```

## Credential Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Initiated: User clicks sign-in
    Initiated --> Authorizing: Popup opened
    Authorizing --> Authorized: User approves
    Authorizing --> Failed: User denies
    Authorized --> Active: Tokens stored
    Active --> Refreshing: Token expires
    Refreshing --> Active: Token refreshed
    Refreshing --> Expired: Refresh fails
    Active --> Revoked: User deletes
    Failed --> [*]
    Expired --> [*]
    Revoked --> [*]
    
    note right of Active: Credentials can be used
    note right of Refreshing: Automatic process
    note right of Revoked: Tokens revoked at provider
```

## OAuth Types Comparison

### User Authentication (SSO) via Supabase

- **Purpose**: Authenticate users to access the AutoGPT platform
- **Provider**: Supabase Auth (currently supports Google SSO)
- **Flow Path**: `/login` → Supabase OAuth → `/auth/callback`
- **Session Storage**: Supabase-managed cookies
- **Token Management**: Automatic by Supabase
- **User Experience**: Single sign-on to the platform

### API Integration Credentials

- **Purpose**: Grant AutoGPT access to user's third-party services
- **Providers**: Examples include GitHub, Google APIs, Notion, and others
  - Full list in `/backend/backend/integrations/providers.py`
  - OAuth handlers in `/backend/backend/integrations/oauth/`
- **Flow Path**: Integration settings → `/api/integrations/{provider}/login` → `/auth/integrations/oauth_callback`
- **Credential Storage**: Encrypted in PostgreSQL
- **Token Management**: Custom refresh logic with mutex locking
- **User Experience**: Connect external services to use in workflows

## Data Flow and Security

### 1. State Token Flow

- **Generation**: Random 32-byte token using `secrets.token_urlsafe()`
- **Storage**: Redis with 10-minute expiration
- **Validation**: Constant-time comparison using `secrets.compare_digest()`
- **Purpose**: CSRF protection and request correlation

### 2. PKCE Implementation

- **Code Verifier**: Random string generated using `secrets.token_urlsafe(128)` (approximately 171 characters when base64url encoded, though RFC 7636 recommends 43-128 characters)
- **Code Challenge**: SHA256 hash of verifier, base64url encoded
- **Storage**: Stored with state token in database (encrypted) with 10-minute expiration
- **Usage**: Enhanced security for public clients (currently used by Twitter provider)

### 3. Credential Storage

- **Structure**:

  ```python
  OAuth2Credentials:
    - id: UUID
    - provider: ProviderName
    - access_token: SecretStr (encrypted)
    - refresh_token: Optional[SecretStr]
    - scopes: List[str]
    - expires_at: Optional[int]
    - username: Optional[str]
  ```

- **Persistence**: PostgreSQL via Prisma ORM
- **Access Control**: User-scoped with mutex locking

### 4. Token Security

- **Storage**: Tokens stored as `SecretStr` type
- **Transport**: HTTPS-only, never logged
- **Refresh**: Automatic refresh 5 minutes before expiry
- **Revocation**: Supported for providers that implement it

## Provider Implementations

### Supported Providers

The platform supports various OAuth providers including GitHub, Google, Notion, Twitter, and others. For the complete list, see:
- `/backend/backend/integrations/providers.py` - All supported providers
- `/backend/backend/integrations/oauth/` - OAuth implementations

### Provider-Specific Security Considerations

- **GitHub**: Supports optional token expiration - tokens may be non-expiring by default
- **Linear**: Returns scopes as space-separated string, requiring special parsing
- **Google**: Requires explicit offline access scope for refresh tokens
- **Twitter**: Uses PKCE for enhanced security on public clients

Each provider handler implements the security measures defined in `BaseOAuthHandler`, ensuring consistent token management and refresh logic across all integrations.

## Security Best Practices

### 1. Frontend Security

- Use popup windows to prevent URL tampering
- Validate state tokens before processing callbacks
- Clear sensitive data from window messages
- Implement timeout for OAuth flows (5 minutes)

### 2. Backend Security

- Store client secrets in environment variables
- Use HTTPS for all OAuth endpoints
- Implement proper scope validation
- Log security events without exposing tokens
- Use database transactions for credential updates

### 3. Token Management

- Refresh tokens proactively (5 minutes before expiry)
- Revoke tokens when credentials are deleted
- Never expose tokens in logs or error messages
- Use constant-time comparison for token validation

## Error Handling

### Common Error Scenarios

1. **Invalid State Token**: 400 Bad Request
2. **Provider Configuration Missing**: 501 Not Implemented
3. **Token Exchange Failure**: 400 Bad Request with hint
4. **Webhook Conflicts**: 409 Conflict, requires confirmation
5. **Credential Not Found**: 404 Not Found

### Error Response Format

```json
{
  "detail": {
    "message": "Human-readable error description",
    "hint": "Actionable suggestion for resolution"
  }
}
```

## Testing Considerations

### Unit Testing

- Mock OAuth providers for flow testing
- Test state token generation and validation
- Verify PKCE implementation
- Test concurrent access scenarios

### Integration Testing

- Use provider sandboxes when available
- Test full OAuth flow with real providers
- Verify token refresh mechanisms
- Test error scenarios and recovery

### Logging Guidelines

- Log flow initiation and completion
- Log errors with context (no tokens)
- Track provider-specific issues
- Monitor for suspicious patterns
