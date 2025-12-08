# AutoGPT OAuth Provider & Credential Broker Design

## Executive Summary

This document outlines a design for AutoGPT to act as:

1. **OAuth Provider**: External platforms (e.g., Lovable) can implement "Sign in with AutoGPT"
2. **Credential Broker**: Handle third-party OAuth (e.g., Google) on behalf of external platforms, keeping tokens secure

This enables a two-step authentication flow:
1. User authenticates with AutoGPT (establishes identity)
2. User authorizes third-party integrations via AutoGPT (credentials never exposed to the calling platform)

## Problem Statement

External platforms building on AutoGPT need access to user integrations but:
- **Security Risk**: Sharing OAuth tokens with external platforms creates attack surface
- **Token Management**: External platforms would need to implement refresh logic, revocation, etc.
- **User Trust**: Users may be reluctant to grant tokens to multiple platforms
- **Compliance**: Token sharing complicates data handling and audit trails

## Proposed Solution

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         External Platform (e.g., Lovable)               │
│                                                                         │
│  ┌──────────────────┐          ┌──────────────────────────────────────┐ │
│  │ "Auth with       │          │ "Sign in with Google via AutoGPT"   │ │
│  │  AutoGPT" Button │          │ Button (opens iframe/popup)         │ │
│  └────────┬─────────┘          └─────────────────┬────────────────────┘ │
│           │                                      │                      │
└───────────┼──────────────────────────────────────┼──────────────────────┘
            │                                      │
            ▼                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                            AutoGPT Platform                               │
│                                                                           │
│  ┌─────────────────────────┐    ┌─────────────────────────────────────┐  │
│  │   OAuth Provider        │    │  Integration OAuth Page             │  │
│  │   (AutoGPT as IdP)      │    │  (Embedded in iframe)               │  │
│  │                         │    │                                     │  │
│  │  - /oauth/authorize     │    │  - Shows "Connect Google" button    │  │
│  │  - /oauth/token         │    │  - Handles full OAuth flow          │  │
│  │  - /oauth/userinfo      │    │  - Returns success/failure to       │  │
│  │                         │    │    parent window                    │  │
│  └────────────┬────────────┘    └──────────────────┬──────────────────┘  │
│               │                                    │                      │
│               ▼                                    ▼                      │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │                    Credential Vault (Existing)                     │   │
│  │                                                                    │   │
│  │  - Encrypted token storage                                         │   │
│  │  - Automatic token refresh                                         │   │
│  │  - Secure credential isolation per user                            │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

## Architecture Components

### Component 1: AutoGPT as OAuth Provider

AutoGPT becomes an OAuth 2.0 Authorization Server that external applications can integrate with.

#### OAuth 2.0 Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/oauth/authorize` | GET | Authorization endpoint - user consent screen |
| `/oauth/token` | POST | Token endpoint - exchange code for tokens |
| `/oauth/userinfo` | GET | UserInfo endpoint - return user profile |
| `/oauth/revoke` | POST | Token revocation endpoint |
| `/.well-known/openid-configuration` | GET | OIDC discovery document |
| `/.well-known/jwks.json` | GET | JSON Web Key Set for token verification |

#### Client Registration Model

```python
class OAuthClient(BaseModel):
    """Registered OAuth client (external application)"""
    client_id: str                    # Unique client identifier
    client_secret: SecretStr          # Client secret (hashed in storage)
    client_name: str                  # Display name
    redirect_uris: list[str]          # Allowed callback URLs
    allowed_scopes: list[str]         # Permitted scopes
    logo_uri: Optional[str]           # For consent screen
    created_at: datetime
    created_by_user_id: str           # AutoGPT user who registered the app

class OAuthClientScopes(str, Enum):
    """Scopes that external apps can request"""
    OPENID = "openid"                 # Basic OIDC - required
    PROFILE = "profile"               # User profile info
    EMAIL = "email"                   # User email
    INTEGRATIONS_READ = "integrations:read"    # List user's integrations
    INTEGRATIONS_USE = "integrations:use"      # Use integrations on behalf of user
```

#### Authorization Flow

```mermaid
sequenceDiagram
    participant User
    participant Lovable as External App (Lovable)
    participant AutoGPT
    participant DB as Database

    User->>Lovable: Click "Sign in with AutoGPT"
    Lovable->>AutoGPT: GET /oauth/authorize?client_id=X&redirect_uri=Y&scope=Z&state=S
    AutoGPT->>User: Show consent screen
    User->>AutoGPT: Approve
    AutoGPT->>DB: Store authorization code
    AutoGPT->>Lovable: Redirect to callback with code
    Lovable->>AutoGPT: POST /oauth/token (code + client_secret)
    AutoGPT->>DB: Validate code, create tokens
    AutoGPT->>Lovable: Return access_token + refresh_token
    Lovable->>AutoGPT: GET /oauth/userinfo
    AutoGPT->>Lovable: Return user profile
```

### Component 2: Integration OAuth Proxy (Credential Broker)

A dedicated page/endpoint for handling third-party OAuth flows on behalf of external applications.

#### Integration OAuth Page

A standalone page that can be embedded in an iframe or opened as a popup:

```
/integrations/oauth-connect?
    provider=google&
    scopes=gmail.readonly,drive&
    client_id=EXTERNAL_APP_CLIENT_ID&
    callback_origin=https://lovable.dev&
    state=EXTERNAL_APP_STATE
```

#### Page Flow

```mermaid
sequenceDiagram
    participant User
    participant Lovable as External App
    participant IFrame as AutoGPT OAuth Page (iframe)
    participant AutoGPT as AutoGPT Backend
    participant Google

    User->>Lovable: Click "Connect Google via AutoGPT"
    Lovable->>IFrame: Open /integrations/oauth-connect in iframe
    IFrame->>User: Show "Connect your Google account" button
    User->>IFrame: Click connect
    IFrame->>AutoGPT: GET /integrations/google/login
    AutoGPT->>IFrame: Return Google OAuth URL
    IFrame->>Google: Popup for Google auth
    User->>Google: Authorize
    Google->>IFrame: Return with code
    IFrame->>AutoGPT: POST /integrations/google/callback
    AutoGPT->>AutoGPT: Store tokens (user never sees them)
    AutoGPT->>IFrame: Return credential_id
    IFrame->>Lovable: postMessage({ success: true, credential_id: "xxx" })
    Lovable->>Lovable: Store credential_id reference
```

#### Security Model

The iframe approach provides several security benefits:

1. **Token Isolation**: The external app never receives actual OAuth tokens
2. **Same-Origin Policy**: External app cannot access iframe contents
3. **Controlled Communication**: Only `postMessage` with validated origins
4. **Credential Reference**: External app only gets a `credential_id` to use via API

### Component 3: Credential Proxy API

External applications use credentials via API calls, never accessing tokens directly.

#### Proxy Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/proxy/{credential_id}/{service_path}` | ANY | Proxy requests using stored credentials |
| `/api/v1/credentials/{credential_id}/status` | GET | Check credential validity |
| `/api/v1/credentials/{credential_id}/refresh` | POST | Force token refresh |

#### Proxy Flow Example

```mermaid
sequenceDiagram
    participant Lovable as External App
    participant AutoGPT as AutoGPT Backend
    participant Google as Google API

    Lovable->>AutoGPT: POST /api/v1/proxy/{cred_id}/gmail/v1/messages
    Note right of AutoGPT: Validates: OAuth token, credential ownership, scope
    AutoGPT->>AutoGPT: Retrieve stored Google token
    AutoGPT->>AutoGPT: Refresh if needed
    AutoGPT->>Google: Forward request with actual token
    Google->>AutoGPT: Response
    AutoGPT->>Lovable: Return response (tokens stripped)
```

## Data Models

### New Database Tables

```prisma
// OAuth Provider Tables
model OAuthClient {
    id              String   @id @default(uuid())
    clientId        String   @unique  // Public identifier
    clientSecretHash String            // Hashed secret
    name            String
    description     String?
    logoUri         String?
    redirectUris    String[]          // JSON array
    allowedScopes   String[]          // JSON array

    createdById     String
    createdBy       User     @relation(fields: [createdById], references: [id])
    createdAt       DateTime @default(now())
    updatedAt       DateTime @updatedAt

    authorizations  OAuthAuthorization[]
}

model OAuthAuthorization {
    id              String   @id @default(uuid())
    userId          String
    user            User     @relation(fields: [userId], references: [id])
    clientId        String
    client          OAuthClient @relation(fields: [clientId], references: [id])

    scopes          String[]          // Granted scopes
    accessToken     String   @unique  // Hashed
    refreshToken    String?  @unique  // Hashed
    accessTokenExpiresAt  DateTime
    refreshTokenExpiresAt DateTime?

    createdAt       DateTime @default(now())
    revokedAt       DateTime?

    @@unique([userId, clientId])
}

model OAuthAuthorizationCode {
    id              String   @id @default(uuid())
    code            String   @unique
    clientId        String
    userId          String
    redirectUri     String
    scopes          String[]
    codeChallenge   String?           // PKCE
    codeChallengeMethod String?       // S256
    expiresAt       DateTime
    usedAt          DateTime?

    @@index([code])
}

// External App Credential Access
model ExternalCredentialGrant {
    id              String   @id @default(uuid())
    credentialId    String            // References user's integration credential
    oauthClientId   String            // Which external app has access
    client          OAuthClient @relation(fields: [oauthClientId], references: [id])

    grantedScopes   String[]          // Which integration scopes are allowed
    grantedAt       DateTime @default(now())
    revokedAt       DateTime?

    @@unique([credentialId, oauthClientId])
}
```

## API Specification

### OAuth Provider Endpoints

#### GET /oauth/authorize

Initiates OAuth flow for external applications.

**Query Parameters:**
- `client_id` (required): Registered client ID
- `redirect_uri` (required): Must match registered URI
- `response_type` (required): Must be `code`
- `scope` (required): Space-separated scopes
- `state` (required): CSRF token from client
- `code_challenge` (optional): PKCE challenge
- `code_challenge_method` (optional): Must be `S256`

**Response:**
- Renders consent page or redirects with error

#### POST /oauth/token

Exchange authorization code for tokens.

**Request Body:**
```json
{
    "grant_type": "authorization_code",
    "code": "AUTH_CODE",
    "redirect_uri": "https://app.example.com/callback",
    "client_id": "CLIENT_ID",
    "client_secret": "CLIENT_SECRET",
    "code_verifier": "PKCE_VERIFIER"  // If PKCE used
}
```

**Response:**
```json
{
    "access_token": "eyJ...",
    "token_type": "Bearer",
    "expires_in": 3600,
    "refresh_token": "dGhpcyBpcyBh...",
    "scope": "openid profile integrations:read"
}
```

#### GET /oauth/userinfo

Get authenticated user's profile.

**Headers:**
- `Authorization: Bearer ACCESS_TOKEN`

**Response:**
```json
{
    "sub": "user_123",
    "name": "John Doe",
    "email": "john@example.com",
    "picture": "https://...",
    "integrations": [  // If integrations:read scope granted
        {
            "provider": "google",
            "credential_id": "cred_abc",
            "scopes": ["gmail.readonly", "drive"],
            "connected_at": "2024-01-15T10:30:00Z"
        }
    ]
}
```

### Integration Proxy Endpoints

#### GET /integrations/oauth-connect

Embeddable OAuth connection page.

**Query Parameters:**
- `provider` (required): Integration provider (google, github, etc.)
- `scopes` (optional): Comma-separated scopes to request
- `client_id` (required): External app's client ID
- `callback_origin` (required): Origin for postMessage
- `state` (optional): Passed back to external app

**Behavior:**
1. Validates `callback_origin` against client's registered domains
2. Renders minimal UI with "Connect [Provider]" button
3. On success/failure, sends postMessage to parent

**postMessage Format:**
```javascript
// Success
{
    type: "autogpt_oauth_result",
    success: true,
    credential_id: "cred_abc123",
    provider: "google",
    scopes: ["gmail.readonly", "drive"],
    state: "original_state"
}

// Failure
{
    type: "autogpt_oauth_result",
    success: false,
    error: "user_cancelled",
    error_description: "User closed the authorization window",
    state: "original_state"
}
```

#### POST /api/v1/proxy/{credential_id}/*

Proxy requests to third-party APIs using stored credentials.

**Headers:**
- `Authorization: Bearer AUTOGPT_ACCESS_TOKEN`

**Path:**
- `credential_id`: The credential reference
- `*`: Service path to proxy (e.g., `gmail/v1/messages`)

**Request:**
- Body and headers forwarded to target API
- `Authorization` header replaced with stored token

**Response:**
- Proxied response from target API
- Sensitive headers stripped

## Security Considerations

### 1. Token Security

| Layer | Mechanism |
|-------|-----------|
| Storage | Tokens encrypted at rest (existing) |
| Transit | HTTPS only, HSTS enabled |
| Access | User + client scoped grants |
| Isolation | External apps never see raw tokens |
| Rotation | Automatic refresh, short-lived access tokens |

### 2. Client Verification

```python
def validate_client_request(
    client_id: str,
    redirect_uri: str,
    origin: Optional[str] = None
) -> OAuthClient:
    client = get_oauth_client(client_id)
    if not client:
        raise InvalidClientError()

    # Validate redirect URI
    if redirect_uri not in client.redirect_uris:
        raise InvalidRedirectUriError()

    # Validate origin for iframe/popup flows
    if origin:
        allowed_origins = [urlparse(uri).origin for uri in client.redirect_uris]
        if origin not in allowed_origins:
            raise InvalidOriginError()

    return client
```

### 3. Scope Restrictions

External apps can only access integrations with explicit user consent:

```python
def check_proxy_access(
    user_id: str,
    client_id: str,
    credential_id: str,
    requested_scope: str
) -> bool:
    grant = get_credential_grant(credential_id, client_id)
    if not grant or grant.revoked_at:
        return False

    # Check if requested operation is within granted scopes
    return requested_scope in grant.granted_scopes
```

### 4. iframe Security Headers

```python
# For /integrations/oauth-connect page
response.headers["X-Frame-Options"] = "ALLOWALL"  # Allow embedding
response.headers["Content-Security-Policy"] = (
    "frame-ancestors 'self' https://*.lovable.dev https://*.registered-domain.com;"
)
```

### 5. Rate Limiting

| Endpoint | Limit | Window |
|----------|-------|--------|
| `/oauth/authorize` | 10 | per minute per IP |
| `/oauth/token` | 20 | per minute per client |
| `/api/v1/proxy/*` | 100 | per minute per credential |

## Implementation Phases

### Phase 1: OAuth Provider Foundation
- [ ] Database schema for OAuth clients and authorizations
- [ ] Client registration API (admin only initially)
- [ ] `/oauth/authorize` endpoint with consent UI
- [ ] `/oauth/token` endpoint with PKCE support
- [ ] `/oauth/userinfo` endpoint
- [ ] OIDC discovery endpoints

### Phase 2: Integration OAuth Page
- [ ] Create `/integrations/oauth-connect` page
- [ ] Minimal, embeddable UI design
- [ ] postMessage communication layer
- [ ] Origin validation against registered clients
- [ ] Error handling and user feedback

### Phase 3: Credential Proxy API
- [ ] Proxy routing infrastructure
- [ ] Request/response sanitization
- [ ] Credential access grants model
- [ ] Audit logging for proxy requests
- [ ] Rate limiting per client/credential

### Phase 4: External Developer Experience
- [ ] Client registration portal
- [ ] Documentation and integration guides
- [ ] SDK/examples for common platforms
- [ ] Webhook notifications for token events

## Example Integration Flow

### For External Platform (Lovable)

```javascript
// 1. Register as OAuth client (one-time setup)
// Done via AutoGPT developer portal

// 2. Implement "Sign in with AutoGPT"
const loginWithAutoGPT = () => {
    const params = new URLSearchParams({
        client_id: 'your_client_id',
        redirect_uri: 'https://yourapp.com/auth/autogpt/callback',
        response_type: 'code',
        scope: 'openid profile integrations:use',
        state: generateCSRFToken()
    });
    window.location.href = `https://autogpt.com/oauth/authorize?${params}`;
};

// 3. Handle callback
app.get('/auth/autogpt/callback', async (req, res) => {
    const { code, state } = req.query;
    validateState(state);

    const tokens = await fetch('https://autogpt.com/oauth/token', {
        method: 'POST',
        body: JSON.stringify({
            grant_type: 'authorization_code',
            code,
            redirect_uri: 'https://yourapp.com/auth/autogpt/callback',
            client_id: CLIENT_ID,
            client_secret: CLIENT_SECRET
        })
    }).then(r => r.json());

    // Store tokens, create session
    createSession(tokens);
});

// 4. Show "Connect Google via AutoGPT" button
const connectGoogleViaAutoGPT = () => {
    const iframe = document.createElement('iframe');
    iframe.src = 'https://autogpt.com/integrations/oauth-connect?' +
        new URLSearchParams({
            provider: 'google',
            scopes: 'gmail.readonly,drive',
            client_id: CLIENT_ID,
            callback_origin: window.location.origin,
            state: 'connect_google_' + Date.now()
        });
    iframe.style.cssText = 'width:400px;height:300px;border:none;';

    // Listen for result
    window.addEventListener('message', (event) => {
        if (event.origin !== 'https://autogpt.com') return;
        if (event.data.type !== 'autogpt_oauth_result') return;

        if (event.data.success) {
            // Store credential_id for later API calls
            saveCredentialReference(event.data.credential_id);
        }

        iframe.remove();
    });

    document.body.appendChild(iframe);
};

// 5. Use Google via AutoGPT proxy
const getGmailMessages = async (credentialId) => {
    const response = await fetch(
        `https://autogpt.com/api/v1/proxy/${credentialId}/gmail/v1/users/me/messages`,
        {
            headers: {
                'Authorization': `Bearer ${autogptAccessToken}`
            }
        }
    );
    return response.json();
};
```

## Open Questions

1. **Client Registration**: Self-service portal or admin-approved only?
2. **Proxy Scope**: Full API passthrough or limited to specific endpoints?
3. **Billing**: Should proxy usage be metered/billed?
4. **Token Lifetime**: How long should external app tokens last?
5. **Revocation UX**: How do users manage which apps have access?

## References

- [OAuth 2.0 RFC 6749](https://tools.ietf.org/html/rfc6749)
- [PKCE RFC 7636](https://tools.ietf.org/html/rfc7636)
- [OpenID Connect Core](https://openid.net/specs/openid-connect-core-1_0.html)
- [Existing OAuth Documentation](./oauth-integration-flow.md)
- [External API Routes](../../backend/backend/server/external/routes/integrations.py)
