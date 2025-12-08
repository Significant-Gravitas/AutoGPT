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
┌─────────────────────────────────────────────────────────────────────────────┐
│                         External Platform (e.g., Lovable)                   │
│                                                                             │
│  ┌──────────────────┐              ┌──────────────────────────────────────┐ │
│  │ "Auth with       │              │ "Connect Google via AutoGPT"        │ │
│  │  AutoGPT" Button │              │ Button (opens popup window)         │ │
│  └────────┬─────────┘              └─────────────────┬────────────────────┘ │
│           │                                          │                      │
└───────────┼──────────────────────────────────────────┼──────────────────────┘
            │                                          │
            │ redirect                                 │ window.open()
            ▼                                          ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                              AutoGPT Platform                                 │
│                                                                               │
│  ┌─────────────────────────┐      ┌─────────────────────────────────────────┐│
│  │   OAuth Provider        │      │  Integration Connect Popup              ││
│  │   (AutoGPT as IdP)      │      │  (Separate window - URL visible)        ││
│  │                         │      │                                         ││
│  │  - /oauth/authorize     │      │  User can verify: autogpt.com/connect   ││
│  │  - /oauth/token         │      │                                         ││
│  │  - /oauth/userinfo      │      │  Shows explicit consent:                ││
│  │                         │      │  "Lovable wants to access your Gmail"   ││
│  │  PKCE Required          │      │                                         ││
│  └────────────┬────────────┘      │  → Opens Google OAuth (also popup)      ││
│               │                   │  → Returns credential grant to opener   ││
│               │                   └──────────────────┬──────────────────────┘│
│               │                                      │                       │
│               ▼                                      ▼                       │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    Credential Vault (Existing)                         │  │
│  │                                                                        │  │
│  │  - Encrypted token storage          - Automatic token refresh          │  │
│  │  - Secure credential isolation      - Comprehensive audit logging      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    Credential Proxy API                                │  │
│  │                                                                        │  │
│  │  - Allowlisted API paths only       - Per-credential scope enforcement │  │
│  │  - Request/response sanitization    - Rate limiting per client         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## Security Architecture

### Why Popup Windows (Not Iframes)

**Iframes are unsuitable for OAuth flows:**

| Issue | Impact |
|-------|--------|
| **OAuth providers block iframes** | Google, GitHub, etc. set `X-Frame-Options: DENY` - their consent screens won't render |
| **Clickjacking** | Malicious sites can overlay invisible elements to hijack authorization clicks |
| **No URL verification** | Users cannot verify they're on `autogpt.com` when embedded |
| **Phishing risk** | Fake consent screens become trivial to create |

**Popup windows are the industry standard** (Stripe Connect, Plaid, Ayrshare):
- User sees real URL bar (can verify `autogpt.com`)
- OAuth provider consent screens work normally
- Established UX pattern users recognize
- Same-origin policy still protects token exchange

### Security Principles

1. **PKCE Required Everywhere**: No client secrets in frontend code
2. **Explicit Scoped Grants**: External apps request specific integration capabilities, not blanket access
3. **Proxy Allowlists**: Only pre-defined API paths can be proxied
4. **Defense in Depth**: Multiple validation layers at each step
5. **Comprehensive Audit Trail**: Every proxy request logged

---

## Architecture Components

### Component 1: AutoGPT as OAuth Provider

AutoGPT becomes an OAuth 2.0 Authorization Server that external applications integrate with.

#### OAuth 2.0 Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/oauth/authorize` | GET | Authorization endpoint - user consent screen |
| `/oauth/token` | POST | Token endpoint - exchange code for tokens |
| `/oauth/userinfo` | GET | UserInfo endpoint - return user profile |
| `/oauth/revoke` | POST | Token revocation endpoint |
| `/.well-known/openid-configuration` | GET | OIDC discovery document |
| `/.well-known/jwks.json` | GET | JSON Web Key Set for token verification |

#### Client Types & PKCE Requirements

| Client Type | Description | Secret Storage | PKCE |
|-------------|-------------|----------------|------|
| **Public** | SPAs, mobile apps, CLI tools | Cannot store secrets | **Required** |
| **Confidential** | Backend servers | Secure server-side | **Required** (defense in depth) |

**PKCE is mandatory for ALL clients** - even confidential ones. This provides:
- Protection against authorization code interception
- Defense in depth if client secret is compromised
- Consistent security model

#### Client Registration Model

```python
class OAuthClientType(str, Enum):
    PUBLIC = "public"           # Cannot securely store secrets
    CONFIDENTIAL = "confidential"  # Server-side apps

class OAuthClient(BaseModel):
    """Registered OAuth client (external application)"""
    client_id: str                      # Unique public identifier
    client_secret_hash: Optional[str]   # Hashed (confidential clients only)
    client_type: OAuthClientType
    client_name: str                    # Display name for consent screen
    client_description: str             # Shown to users
    logo_uri: Optional[str]             # For consent screen

    # Security configuration
    redirect_uris: list[str]            # Strict allowlist
    allowed_scopes: list[str]           # Maximum scopes this client can request
    allowed_integration_providers: list[str]  # Which integrations (google, github, etc.)

    # Metadata
    privacy_policy_uri: Optional[str]
    terms_of_service_uri: Optional[str]
    contacts: list[str]                 # Developer contact emails

    # Audit
    created_at: datetime
    created_by_user_id: str
    approved_at: Optional[datetime]     # Admin approval timestamp
    approved_by: Optional[str]

class OAuthScopes(str, Enum):
    """Scopes that external apps can request from AutoGPT"""
    # OpenID Connect standard scopes
    OPENID = "openid"                   # Required - basic OIDC
    PROFILE = "profile"                 # User profile info
    EMAIL = "email"                     # User email

    # AutoGPT-specific scopes
    INTEGRATIONS_LIST = "integrations:list"      # List connected integrations
    INTEGRATIONS_CONNECT = "integrations:connect" # Initiate new integration OAuth
    INTEGRATIONS_USE = "integrations:use"        # Use integrations via proxy
```

#### Authorization Flow (with PKCE)

```mermaid
sequenceDiagram
    participant User
    participant Lovable as External App
    participant AutoGPT
    participant DB as Database

    Note over Lovable: Generate PKCE pair
    Lovable->>Lovable: code_verifier = random(43-128 chars)
    Lovable->>Lovable: code_challenge = BASE64URL(SHA256(code_verifier))

    User->>Lovable: Click "Sign in with AutoGPT"
    Lovable->>AutoGPT: GET /oauth/authorize?<br/>client_id=X&redirect_uri=Y&<br/>scope=Z&state=S&<br/>code_challenge=C&<br/>code_challenge_method=S256

    AutoGPT->>AutoGPT: Validate client_id, redirect_uri
    AutoGPT->>User: Show consent screen
    User->>AutoGPT: Approve
    AutoGPT->>DB: Store auth code + code_challenge
    AutoGPT->>Lovable: Redirect with code + state

    Lovable->>AutoGPT: POST /oauth/token<br/>code + code_verifier + client_id
    AutoGPT->>AutoGPT: Verify SHA256(code_verifier) == stored challenge
    AutoGPT->>DB: Validate code, mark used, create tokens
    AutoGPT->>Lovable: Return access_token + refresh_token
```

---

### Component 2: Integration Connect (Credential Broker)

A **popup window** flow for connecting third-party integrations through AutoGPT.

#### Why This Architecture

```
┌─────────────────────────────────────┐
│ Lovable App (opener window)         │
│                                     │
│ [Connect Google via AutoGPT]        │
│         │                           │
│         │ window.open()             │
│         ▼                           │
│   ┌─────────────────────────────────────────────────────┐
│   │                                                     │
│   │  AutoGPT Popup Window                               │
│   │  ════════════════════════════════════════════════   │
│   │  🔒 autogpt.com/connect/google                     │ ← User can verify URL
│   │  ════════════════════════════════════════════════   │
│   │                                                     │
│   │  ┌─────────────────────────────────────────────┐   │
│   │  │         [AutoGPT Logo]                      │   │
│   │  │                                             │   │
│   │  │  "Lovable" wants to access your             │   │
│   │  │  Google account through AutoGPT            │   │
│   │  │                                             │   │
│   │  │  This will allow Lovable to:                │   │
│   │  │  ✓ Read your Gmail messages                 │   │ ← Explicit scope list
│   │  │  ✓ View your Google Drive files             │   │
│   │  │                                             │   │
│   │  │  Lovable will NOT have access to:           │   │
│   │  │  ✗ Your Google password                     │   │
│   │  │  ✗ Other Google services                    │   │
│   │  │  ✗ The actual OAuth tokens                  │   │
│   │  │                                             │   │
│   │  │  [Cancel]  [Connect with Google]            │   │
│   │  └─────────────────────────────────────────────┘   │
│   │                                                     │
│   └─────────────────────────────────────────────────────┘
│                                     │
│   Waits for postMessage...          │
│   Receives: { credential_id, scopes }
│   Never receives: actual tokens     │
└─────────────────────────────────────┘
```

#### Scoped Credential Grants

External apps don't get blanket access. They request specific **integration scopes**:

```python
class IntegrationScope(str, Enum):
    """Fine-grained scopes for each integration provider"""

    # Google
    GOOGLE_GMAIL_READ = "google:gmail.readonly"
    GOOGLE_GMAIL_SEND = "google:gmail.send"
    GOOGLE_DRIVE_READ = "google:drive.readonly"
    GOOGLE_DRIVE_WRITE = "google:drive.file"
    GOOGLE_CALENDAR_READ = "google:calendar.readonly"
    GOOGLE_CALENDAR_WRITE = "google:calendar.events"
    GOOGLE_SHEETS_READ = "google:sheets.readonly"
    GOOGLE_SHEETS_WRITE = "google:spreadsheets"

    # GitHub
    GITHUB_REPOS_READ = "github:repo.read"
    GITHUB_REPOS_WRITE = "github:repo.write"
    GITHUB_ISSUES_READ = "github:issues.read"
    GITHUB_ISSUES_WRITE = "github:issues.write"

    # ... etc for each provider

class CredentialGrant(BaseModel):
    """Records what an external app can do with a user's credential"""
    id: str
    user_id: str
    client_id: str                      # Which external app
    credential_id: str                  # Which user credential

    granted_scopes: list[IntegrationScope]  # Specific capabilities granted

    created_at: datetime
    expires_at: Optional[datetime]      # Optional time limit
    revoked_at: Optional[datetime]

    # Audit trail
    grant_ip: str
    grant_user_agent: str
```

#### Integration Connect Flow

```mermaid
sequenceDiagram
    participant User
    participant Lovable as External App
    participant Popup as AutoGPT Popup
    participant Backend as AutoGPT Backend
    participant Google

    User->>Lovable: Click "Connect Google via AutoGPT"

    Note over Lovable: Generate PKCE + state
    Lovable->>Popup: window.open(/connect/google?<br/>client_id=X&scopes=gmail.readonly,drive&<br/>state=S&code_challenge=C)

    Popup->>Backend: Validate client, scopes
    Backend->>Backend: Check client allowed these scopes
    Popup->>User: Show consent screen with explicit scope list

    User->>Popup: Click "Connect with Google"
    Popup->>Google: Redirect to Google OAuth
    User->>Google: Authorize
    Google->>Popup: Redirect with Google auth code

    Popup->>Backend: Exchange Google code for tokens
    Backend->>Backend: Store Google tokens (encrypted)
    Backend->>Backend: Create CredentialGrant for client
    Backend->>Popup: Return grant_id

    Popup->>Lovable: postMessage({ success, grant_id, scopes })
    Popup->>Popup: window.close()

    Note over Lovable: Store grant_id for API calls
```

#### postMessage Security

```typescript
// AutoGPT popup sends to opener
interface OAuthResultMessage {
    type: "autogpt:oauth_result";       // Namespaced type
    nonce: string;                       // Must match original request
    success: boolean;
    grant_id?: string;                   // NOT credential_id - it's the grant
    granted_scopes?: string[];
    error?: string;
    error_description?: string;
}

// Security validations in popup before sending
function sendResultToOpener(result: OAuthResultMessage) {
    // Only send to the specific origin that opened us
    const allowedOrigin = new URL(window.opener.location.href).origin;

    // Validate against registered redirect URIs
    if (!isRegisteredOrigin(allowedOrigin)) {
        console.error("Opener origin not in registered URIs");
        return;
    }

    window.opener.postMessage(result, allowedOrigin);
}

// External app validates received messages
window.addEventListener("message", (event) => {
    // Strict origin check
    if (event.origin !== "https://autogpt.com") return;

    // Type check
    if (event.data?.type !== "autogpt:oauth_result") return;

    // Nonce check (prevents replay)
    if (event.data.nonce !== expectedNonce) return;

    // Process result...
});
```

---

### Component 3: Credential Proxy API

External apps use credentials via a **controlled proxy** - never direct token access.

#### Proxy Allowlists

**Not all API paths are proxied.** Each integration defines allowed operations:

```python
class ProxyAllowlist:
    """Defines what API paths can be proxied for each integration"""

    GOOGLE_GMAIL = {
        "gmail.readonly": [
            AllowedPath("GET", "/gmail/v1/users/me/messages"),
            AllowedPath("GET", "/gmail/v1/users/me/messages/{message_id}"),
            AllowedPath("GET", "/gmail/v1/users/me/threads"),
            AllowedPath("GET", "/gmail/v1/users/me/labels"),
        ],
        "gmail.send": [
            AllowedPath("POST", "/gmail/v1/users/me/messages/send"),
        ],
    }

    GOOGLE_DRIVE = {
        "drive.readonly": [
            AllowedPath("GET", "/drive/v3/files"),
            AllowedPath("GET", "/drive/v3/files/{file_id}"),
            # Notably MISSING: /drive/v3/files/{file_id}/export (data exfil risk)
        ],
        "drive.file": [
            AllowedPath("POST", "/drive/v3/files"),
            AllowedPath("PATCH", "/drive/v3/files/{file_id}"),
            # Notably MISSING: DELETE operations
        ],
    }

    GITHUB = {
        "repo.read": [
            AllowedPath("GET", "/repos/{owner}/{repo}"),
            AllowedPath("GET", "/repos/{owner}/{repo}/contents/{path}"),
        ],
        "issues.read": [
            AllowedPath("GET", "/repos/{owner}/{repo}/issues"),
            AllowedPath("GET", "/repos/{owner}/{repo}/issues/{issue_number}"),
        ],
        "issues.write": [
            AllowedPath("POST", "/repos/{owner}/{repo}/issues"),
            AllowedPath("PATCH", "/repos/{owner}/{repo}/issues/{issue_number}"),
        ],
    }

@dataclass
class AllowedPath:
    method: str
    path_pattern: str  # Supports {param} placeholders

    # Optional restrictions
    max_body_size: int = 1_000_000  # 1MB default
    blocked_body_fields: list[str] = field(default_factory=list)
    rate_limit: Optional[int] = None  # Override default rate limit
```

#### Proxy Validation Flow

```python
async def proxy_request(
    request: Request,
    grant_id: str,
    service_path: str,
    autogpt_token: str,  # From Authorization header
) -> Response:
    """
    Secure proxy with multiple validation layers.
    """

    # Layer 1: Authenticate the external app's AutoGPT token
    token_claims = verify_autogpt_token(autogpt_token)
    client_id = token_claims["client_id"]
    user_id = token_claims["sub"]

    # Layer 2: Validate the grant exists and is active
    grant = await get_credential_grant(grant_id)
    if not grant:
        raise HTTPException(404, "Grant not found")
    if grant.revoked_at:
        raise HTTPException(403, "Grant has been revoked")
    if grant.expires_at and grant.expires_at < datetime.utcnow():
        raise HTTPException(403, "Grant has expired")
    if grant.user_id != user_id:
        raise HTTPException(403, "Grant belongs to different user")
    if grant.client_id != client_id:
        raise HTTPException(403, "Grant belongs to different client")

    # Layer 3: Check if this path is allowed for the granted scopes
    credential = await get_credential(grant.credential_id)
    provider = credential.provider
    allowed_path = find_allowed_path(
        provider=provider,
        granted_scopes=grant.granted_scopes,
        method=request.method,
        path=service_path,
    )
    if not allowed_path:
        await audit_log.blocked_request(
            grant_id=grant_id,
            client_id=client_id,
            path=service_path,
            reason="path_not_allowed",
        )
        raise HTTPException(403, f"Path not allowed for granted scopes")

    # Layer 4: Validate request body (if applicable)
    if request.body:
        body = await request.json()
        if len(request.body) > allowed_path.max_body_size:
            raise HTTPException(413, "Request body too large")
        for blocked_field in allowed_path.blocked_body_fields:
            if blocked_field in body:
                raise HTTPException(
                    400,
                    f"Field '{blocked_field}' not allowed in request"
                )

    # Layer 5: Rate limiting
    rate_limit = allowed_path.rate_limit or DEFAULT_RATE_LIMIT
    if not await check_rate_limit(grant_id, rate_limit):
        raise HTTPException(429, "Rate limit exceeded")

    # Layer 6: Get actual token (auto-refresh if needed)
    access_token = await get_refreshed_token(grant.credential_id)

    # Layer 7: Make proxied request
    target_url = build_target_url(provider, service_path)
    proxied_response = await http_client.request(
        method=request.method,
        url=target_url,
        headers={
            "Authorization": f"Bearer {access_token}",
            # Forward safe headers only
            "Content-Type": request.headers.get("Content-Type"),
            "Accept": request.headers.get("Accept"),
        },
        content=request.body if request.body else None,
    )

    # Layer 8: Sanitize response
    sanitized_headers = {
        k: v for k, v in proxied_response.headers.items()
        if k.lower() not in BLOCKED_RESPONSE_HEADERS
    }

    # Layer 9: Audit log
    await audit_log.proxied_request(
        grant_id=grant_id,
        client_id=client_id,
        user_id=user_id,
        provider=provider,
        method=request.method,
        path=service_path,
        status_code=proxied_response.status_code,
        response_size=len(proxied_response.content),
    )

    return Response(
        content=proxied_response.content,
        status_code=proxied_response.status_code,
        headers=sanitized_headers,
    )

BLOCKED_RESPONSE_HEADERS = {
    "set-cookie",           # Don't leak cookies
    "www-authenticate",     # Don't leak auth challenges
    "x-oauth-scopes",       # Don't leak token scopes
    "x-ratelimit-*",        # Don't leak rate limit info
}
```

#### SSRF Prevention

```python
class ProxyTargetValidator:
    """Prevents SSRF attacks via the proxy"""

    # Only these base URLs are valid proxy targets
    ALLOWED_TARGETS = {
        "google": [
            "https://www.googleapis.com",
            "https://gmail.googleapis.com",
            "https://sheets.googleapis.com",
        ],
        "github": [
            "https://api.github.com",
        ],
        "notion": [
            "https://api.notion.com",
        ],
    }

    @classmethod
    def validate_target(cls, provider: str, path: str) -> str:
        """Returns full URL or raises if invalid"""

        # Path traversal check
        if ".." in path or path.startswith("/"):
            raise ValueError("Invalid path")

        # Must be known provider
        if provider not in cls.ALLOWED_TARGETS:
            raise ValueError(f"Unknown provider: {provider}")

        # Build URL (always first allowed base)
        base_url = cls.ALLOWED_TARGETS[provider][0]
        full_url = f"{base_url}/{path}"

        # Validate final URL is still in allowed list
        parsed = urlparse(full_url)
        if not any(
            full_url.startswith(allowed)
            for allowed in cls.ALLOWED_TARGETS[provider]
        ):
            raise ValueError("URL not in allowed targets")

        return full_url
```

---

### Component 4: Audit Logging

Every action through the credential broker is logged for security and compliance.

#### Audit Event Types

```python
class AuditEventType(str, Enum):
    # OAuth Provider events
    CLIENT_REGISTERED = "client.registered"
    CLIENT_APPROVED = "client.approved"
    CLIENT_SUSPENDED = "client.suspended"

    AUTHORIZATION_REQUESTED = "auth.requested"
    AUTHORIZATION_GRANTED = "auth.granted"
    AUTHORIZATION_DENIED = "auth.denied"

    TOKEN_ISSUED = "token.issued"
    TOKEN_REFRESHED = "token.refreshed"
    TOKEN_REVOKED = "token.revoked"

    # Credential Broker events
    INTEGRATION_CONNECT_STARTED = "integration.connect.started"
    INTEGRATION_CONNECT_COMPLETED = "integration.connect.completed"
    INTEGRATION_CONNECT_FAILED = "integration.connect.failed"

    GRANT_CREATED = "grant.created"
    GRANT_REVOKED = "grant.revoked"
    GRANT_EXPIRED = "grant.expired"

    # Proxy events
    PROXY_REQUEST = "proxy.request"
    PROXY_BLOCKED = "proxy.blocked"
    PROXY_ERROR = "proxy.error"
    PROXY_RATE_LIMITED = "proxy.rate_limited"

class AuditLog(BaseModel):
    id: str
    timestamp: datetime
    event_type: AuditEventType

    # Who
    user_id: Optional[str]
    client_id: Optional[str]

    # What
    grant_id: Optional[str]
    credential_id: Optional[str]  # Stored but never exposed in queries

    # Context
    ip_address: str
    user_agent: str

    # Details (event-specific)
    details: dict

    # For proxy events
    provider: Optional[str]
    method: Optional[str]
    path: Optional[str]
    status_code: Optional[int]
    response_time_ms: Optional[int]

# Example audit queries for security monitoring
async def detect_suspicious_activity():
    """Run periodically to detect potential abuse"""

    # High volume from single client
    high_volume = await audit_db.query("""
        SELECT client_id, COUNT(*) as count
        FROM audit_logs
        WHERE event_type = 'proxy.request'
        AND timestamp > NOW() - INTERVAL '1 hour'
        GROUP BY client_id
        HAVING COUNT(*) > 10000
    """)

    # Many blocked requests (probing)
    probing = await audit_db.query("""
        SELECT client_id, COUNT(*) as blocked_count
        FROM audit_logs
        WHERE event_type = 'proxy.blocked'
        AND timestamp > NOW() - INTERVAL '1 hour'
        GROUP BY client_id
        HAVING COUNT(*) > 100
    """)

    # Unusual path patterns
    unusual_paths = await audit_db.query("""
        SELECT client_id, path, COUNT(*) as count
        FROM audit_logs
        WHERE event_type = 'proxy.request'
        AND path LIKE '%..%' OR path LIKE '%/etc/%'
        AND timestamp > NOW() - INTERVAL '24 hours'
        GROUP BY client_id, path
    """)

    return {
        "high_volume_clients": high_volume,
        "probing_clients": probing,
        "unusual_path_clients": unusual_paths,
    }
```

---

## Data Models

### Database Schema

```prisma
// ============================================
// OAuth Provider Tables
// ============================================

model OAuthClient {
    id                String   @id @default(uuid())
    clientId          String   @unique  // Public identifier (e.g., "lvbl_abc123")
    clientSecretHash  String?           // Argon2 hash (confidential clients only)
    clientType        String            // "public" | "confidential"

    name              String            // "Lovable"
    description       String            // Shown on consent screen
    logoUri           String?
    privacyPolicyUri  String?
    termsOfServiceUri String?
    contacts          String[]          // Developer emails

    redirectUris      String[]          // Strict allowlist
    allowedScopes     String[]          // Max scopes client can request
    allowedProviders  String[]          // Which integrations (google, github, etc.)

    // Admin approval workflow
    status            String   @default("pending")  // pending, approved, suspended
    createdById       String
    createdBy         User     @relation("ClientCreator", fields: [createdById], references: [id])
    createdAt         DateTime @default(now())
    approvedAt        DateTime?
    approvedById      String?
    approvedBy        User?    @relation("ClientApprover", fields: [approvedById], references: [id])
    suspendedAt       DateTime?
    suspendedReason   String?

    // Relations
    authorizations    OAuthAuthorization[]
    grants            CredentialGrant[]

    @@index([clientId])
    @@index([status])
}

model OAuthAuthorization {
    id                    String   @id @default(uuid())
    userId                String
    user                  User     @relation(fields: [userId], references: [id])
    clientId              String
    client                OAuthClient @relation(fields: [clientId], references: [id])

    scopes                String[]

    // Tokens (hashed - we store hash, return plaintext once)
    accessTokenHash       String   @unique
    refreshTokenHash      String?  @unique
    accessTokenExpiresAt  DateTime
    refreshTokenExpiresAt DateTime?

    createdAt             DateTime @default(now())
    lastUsedAt            DateTime?
    revokedAt             DateTime?

    @@unique([userId, clientId])
    @@index([accessTokenHash])
    @@index([refreshTokenHash])
}

model OAuthAuthorizationCode {
    id                  String   @id @default(uuid())
    code                String   @unique  // Hashed

    clientId            String
    userId              String
    redirectUri         String
    scopes              String[]

    // PKCE (required)
    codeChallenge       String
    codeChallengeMethod String   @default("S256")

    expiresAt           DateTime  // Short-lived (10 minutes)
    usedAt              DateTime?

    @@index([code])
}

// ============================================
// Credential Broker Tables
// ============================================

model CredentialGrant {
    id              String   @id @default(uuid())

    userId          String
    user            User     @relation(fields: [userId], references: [id])
    clientId        String
    client          OAuthClient @relation(fields: [clientId], references: [id])
    credentialId    String   // References user's integration credential

    // Fine-grained scopes (e.g., ["google:gmail.readonly", "google:drive.readonly"])
    grantedScopes   String[]

    // Lifecycle
    createdAt       DateTime @default(now())
    expiresAt       DateTime?
    revokedAt       DateTime?
    lastUsedAt      DateTime?

    // Audit metadata
    grantIp         String
    grantUserAgent  String

    @@unique([userId, clientId, credentialId])
    @@index([userId])
    @@index([clientId])
}

// ============================================
// Audit Tables
// ============================================

model AuditLog {
    id              String   @id @default(uuid())
    timestamp       DateTime @default(now())
    eventType       String

    // Actors
    userId          String?
    clientId        String?

    // Resources
    grantId         String?
    credentialId    String?  // For internal use only, never exposed

    // Request context
    ipAddress       String
    userAgent       String

    // Event details (JSON)
    details         Json

    // For proxy events specifically
    provider        String?
    method          String?
    path            String?
    statusCode      Int?
    responseTimeMs  Int?
    requestSize     Int?
    responseSize    Int?

    @@index([timestamp])
    @@index([eventType])
    @@index([userId])
    @@index([clientId])
    @@index([grantId])
}
```

---

## API Specification

### OAuth Provider Endpoints

#### GET /oauth/authorize

Initiates OAuth flow. **PKCE required.**

**Query Parameters:**
| Parameter | Required | Description |
|-----------|----------|-------------|
| `client_id` | Yes | Registered client ID |
| `redirect_uri` | Yes | Must exactly match registered URI |
| `response_type` | Yes | Must be `code` |
| `scope` | Yes | Space-separated scopes |
| `state` | Yes | CSRF token (min 32 chars recommended) |
| `code_challenge` | **Yes** | BASE64URL(SHA256(code_verifier)) |
| `code_challenge_method` | Yes | Must be `S256` |

**Success Response:**
Redirects to consent page, then to `redirect_uri` with:
```
https://app.example.com/callback?code=AUTH_CODE&state=ORIGINAL_STATE
```

**Error Response:**
```
https://app.example.com/callback?error=invalid_request&error_description=...&state=ORIGINAL_STATE
```

#### POST /oauth/token

Exchange authorization code for tokens.

**Request Body:**
```json
{
    "grant_type": "authorization_code",
    "code": "AUTH_CODE",
    "redirect_uri": "https://app.example.com/callback",
    "client_id": "CLIENT_ID",
    "code_verifier": "ORIGINAL_CODE_VERIFIER"
}
```

For confidential clients, also include:
```json
{
    "client_secret": "CLIENT_SECRET"
}
```

**Success Response:**
```json
{
    "access_token": "agpt_abc123...",
    "token_type": "Bearer",
    "expires_in": 3600,
    "refresh_token": "agpt_ref_xyz789...",
    "scope": "openid profile integrations:use"
}
```

#### POST /oauth/token (Refresh)

```json
{
    "grant_type": "refresh_token",
    "refresh_token": "agpt_ref_xyz789...",
    "client_id": "CLIENT_ID"
}
```

### Integration Connect Endpoints

#### GET /connect/{provider}

Opens the integration connection popup.

**Query Parameters:**
| Parameter | Required | Description |
|-----------|----------|-------------|
| `client_id` | Yes | Registered client ID |
| `scopes` | Yes | Comma-separated integration scopes |
| `state` | Yes | Passed back to opener |
| `code_challenge` | Yes | PKCE challenge |
| `nonce` | Yes | Unique per request, returned in postMessage |

**Example:**
```
/connect/google?client_id=lvbl_abc&scopes=gmail.readonly,drive.readonly&state=xyz&code_challenge=ABC&nonce=123
```

**Behavior:**
1. Validates client is approved and scopes are allowed
2. Renders consent screen showing exactly what's being requested
3. On user approval, initiates Google OAuth
4. On completion, sends postMessage to opener with grant_id
5. Closes popup

### Proxy Endpoints

#### ANY /api/v1/proxy/{grant_id}/{path}

Proxy requests to third-party APIs.

**Headers:**
```
Authorization: Bearer AUTOGPT_ACCESS_TOKEN
Content-Type: application/json (if applicable)
```

**Path Parameters:**
- `grant_id`: The credential grant ID (from connect flow)
- `path`: Service path (e.g., `gmail/v1/users/me/messages`)

**Example:**
```bash
curl -X GET \
  "https://autogpt.com/api/v1/proxy/grant_abc123/gmail/v1/users/me/messages?maxResults=10" \
  -H "Authorization: Bearer agpt_xyz789"
```

**Response:**
- Proxied response from target API
- HTTP status preserved
- Sensitive headers stripped

**Error Responses:**

| Status | Error | Description |
|--------|-------|-------------|
| 401 | `invalid_token` | AutoGPT token invalid/expired |
| 403 | `grant_revoked` | User revoked this grant |
| 403 | `path_not_allowed` | Path not in allowlist for granted scopes |
| 404 | `grant_not_found` | Grant doesn't exist |
| 429 | `rate_limited` | Too many requests |
| 502 | `upstream_error` | Target API returned error |

---

## User Experience

### Consent Screen Design

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                    [AutoGPT Logo]                           │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│     [Lovable Logo]                                          │
│     Lovable wants to access your account                    │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│     This will allow Lovable to:                             │
│                                                             │
│     ✓ View your basic profile information                   │
│     ✓ See your email address                                │
│     ✓ Connect to third-party services on your behalf        │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│     By continuing, you agree to Lovable's                   │
│     Terms of Service and Privacy Policy                     │
│                                                             │
│     ┌─────────────┐       ┌─────────────────────┐          │
│     │   Cancel    │       │   Allow Access      │          │
│     └─────────────┘       └─────────────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Integration Connect Screen

```
┌─────────────────────────────────────────────────────────────┐
│  🔒 autogpt.com/connect/google                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    [AutoGPT Logo]                           │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│     Connect your Google account                             │
│     for use with Lovable                                    │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│     Lovable is requesting access to:                        │
│                                                             │
│     📧 Gmail                                                │
│        • Read your email messages                           │
│        • View email metadata and labels                     │
│                                                             │
│     📁 Google Drive                                         │
│        • View files in your Drive                           │
│        • Download file contents                             │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│     🔒 Security note:                                       │
│     • Lovable will NOT receive your Google password         │
│     • Lovable will NOT receive your Google tokens           │
│     • You can revoke access anytime in AutoGPT settings     │
│                                                             │
│     ┌─────────────┐    ┌────────────────────────────┐      │
│     │   Cancel    │    │  Continue with Google  →   │      │
│     └─────────────┘    └────────────────────────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### User Settings: Manage Connected Apps

```
┌─────────────────────────────────────────────────────────────┐
│  Settings > Connected Applications                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Applications with access to your account:                  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ [Lovable Logo]  Lovable                              │   │
│  │                                                      │   │
│  │ Permissions:                                         │   │
│  │ • View profile and email                             │   │
│  │ • Use integrations on your behalf                    │   │
│  │                                                      │   │
│  │ Connected integrations:                              │   │
│  │ • Google (Gmail read, Drive read)                    │   │
│  │ • GitHub (Repos read)                                │   │
│  │                                                      │   │
│  │ Connected: Jan 15, 2025                              │   │
│  │ Last used: 2 hours ago                               │   │
│  │                                                      │   │
│  │ [Manage Permissions]  [Revoke Access]                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Example Integration

### For External Platform (Lovable)

```typescript
// ============================================
// 1. Sign in with AutoGPT (PKCE flow)
// ============================================

function generatePKCE(): { verifier: string; challenge: string } {
    const verifier = crypto.randomBytes(32).toString('base64url');
    const challenge = crypto
        .createHash('sha256')
        .update(verifier)
        .digest('base64url');
    return { verifier, challenge };
}

async function loginWithAutoGPT() {
    const pkce = generatePKCE();
    const state = crypto.randomBytes(16).toString('hex');

    // Store for callback validation
    sessionStorage.setItem('autogpt_pkce_verifier', pkce.verifier);
    sessionStorage.setItem('autogpt_state', state);

    const params = new URLSearchParams({
        client_id: AUTOGPT_CLIENT_ID,
        redirect_uri: `${window.location.origin}/auth/autogpt/callback`,
        response_type: 'code',
        scope: 'openid profile email integrations:use',
        state,
        code_challenge: pkce.challenge,
        code_challenge_method: 'S256',
    });

    window.location.href = `https://autogpt.com/oauth/authorize?${params}`;
}

// Callback handler
async function handleAutoGPTCallback(code: string, state: string) {
    // Validate state
    const expectedState = sessionStorage.getItem('autogpt_state');
    if (state !== expectedState) {
        throw new Error('Invalid state - possible CSRF attack');
    }

    const verifier = sessionStorage.getItem('autogpt_pkce_verifier');

    // Exchange code for tokens (do this server-side for confidential clients)
    const response = await fetch('https://autogpt.com/oauth/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            grant_type: 'authorization_code',
            code,
            redirect_uri: `${window.location.origin}/auth/autogpt/callback`,
            client_id: AUTOGPT_CLIENT_ID,
            code_verifier: verifier,
        }),
    });

    const tokens = await response.json();
    // Store tokens securely...
}

// ============================================
// 2. Connect Google via AutoGPT (popup flow)
// ============================================

function connectGoogleViaAutoGPT(): Promise<{ grantId: string; scopes: string[] }> {
    return new Promise((resolve, reject) => {
        const pkce = generatePKCE();
        const state = crypto.randomBytes(16).toString('hex');
        const nonce = crypto.randomBytes(16).toString('hex');

        // Store verifier for potential future use
        sessionStorage.setItem(`autogpt_connect_${nonce}`, pkce.verifier);

        const params = new URLSearchParams({
            client_id: AUTOGPT_CLIENT_ID,
            scopes: 'gmail.readonly,drive.readonly',
            state,
            code_challenge: pkce.challenge,
            nonce,
        });

        // Open popup
        const popup = window.open(
            `https://autogpt.com/connect/google?${params}`,
            'autogpt_connect',
            'width=500,height=700,popup=yes'
        );

        if (!popup) {
            reject(new Error('Popup blocked'));
            return;
        }

        // Set timeout
        const timeout = setTimeout(() => {
            cleanup();
            reject(new Error('Connection timed out'));
        }, 5 * 60 * 1000); // 5 minutes

        // Listen for result
        function handleMessage(event: MessageEvent) {
            // Strict origin check
            if (event.origin !== 'https://autogpt.com') return;

            const data = event.data;
            if (data?.type !== 'autogpt:oauth_result') return;

            // Nonce check
            if (data.nonce !== nonce) return;

            cleanup();

            if (data.success) {
                resolve({
                    grantId: data.grant_id,
                    scopes: data.granted_scopes,
                });
            } else {
                reject(new Error(data.error_description || data.error));
            }
        }

        function cleanup() {
            clearTimeout(timeout);
            window.removeEventListener('message', handleMessage);
            sessionStorage.removeItem(`autogpt_connect_${nonce}`);
            if (popup && !popup.closed) {
                popup.close();
            }
        }

        window.addEventListener('message', handleMessage);

        // Also handle popup close
        const pollTimer = setInterval(() => {
            if (popup.closed) {
                clearInterval(pollTimer);
                cleanup();
                reject(new Error('User closed the window'));
            }
        }, 500);
    });
}

// ============================================
// 3. Use Google via AutoGPT proxy
// ============================================

async function getGmailMessages(grantId: string, autogptToken: string) {
    const response = await fetch(
        `https://autogpt.com/api/v1/proxy/${grantId}/gmail/v1/users/me/messages?maxResults=10`,
        {
            headers: {
                'Authorization': `Bearer ${autogptToken}`,
            },
        }
    );

    if (!response.ok) {
        const error = await response.json();
        if (error.error === 'grant_revoked') {
            // User revoked access - prompt to reconnect
            await promptReconnect();
        }
        throw new Error(error.error_description || error.error);
    }

    return response.json();
}

async function sendGmailMessage(grantId: string, autogptToken: string, message: object) {
    // This will fail if user only granted gmail.readonly
    const response = await fetch(
        `https://autogpt.com/api/v1/proxy/${grantId}/gmail/v1/users/me/messages/send`,
        {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${autogptToken}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(message),
        }
    );

    if (response.status === 403) {
        const error = await response.json();
        if (error.error === 'path_not_allowed') {
            // Need additional scopes - prompt user
            await promptAdditionalScopes(['gmail.send']);
        }
    }

    return response.json();
}
```

---

## Rate Limiting

| Endpoint | Limit | Window | Scope |
|----------|-------|--------|-------|
| `GET /oauth/authorize` | 10 | per minute | per IP |
| `POST /oauth/token` | 20 | per minute | per client |
| `GET /connect/{provider}` | 10 | per minute | per client + user |
| `* /api/v1/proxy/*` | 100 | per minute | per grant |
| `* /api/v1/proxy/*` | 1000 | per hour | per grant |
| `* /api/v1/proxy/*` | 10000 | per day | per client |

---

## Implementation Phases

### Phase 1: OAuth Provider Foundation
- [ ] Database schema for clients, authorizations, codes
- [ ] Client registration (admin-only initially)
- [ ] `/oauth/authorize` with consent UI
- [ ] `/oauth/token` with mandatory PKCE
- [ ] `/oauth/userinfo` endpoint
- [ ] OIDC discovery endpoints
- [ ] Basic rate limiting

### Phase 2: Integration Connect Popup
- [ ] `/connect/{provider}` popup page
- [ ] Consent UI with explicit scope display
- [ ] postMessage communication with nonce validation
- [ ] CredentialGrant model and creation
- [ ] Error handling and user feedback

### Phase 3: Credential Proxy API
- [ ] Proxy routing infrastructure
- [ ] Allowlist configuration per provider/scope
- [ ] Request validation pipeline
- [ ] SSRF prevention
- [ ] Response sanitization
- [ ] Rate limiting per grant

### Phase 4: Audit & Monitoring
- [ ] Comprehensive audit logging
- [ ] Suspicious activity detection
- [ ] Admin dashboard for monitoring
- [ ] Alerting for anomalies

### Phase 5: User Experience
- [ ] User settings page for managing connected apps
- [ ] Grant revocation flow
- [ ] Email notifications for new connections
- [ ] Scope upgrade requests

### Phase 6: Developer Experience
- [ ] Self-service client registration portal
- [ ] Documentation and guides
- [ ] SDK for common platforms (JS, Python)
- [ ] Webhook notifications for token events

---

## Security Checklist

### OAuth Provider
- [ ] PKCE mandatory for all clients
- [ ] Authorization codes single-use and short-lived (10 min)
- [ ] Tokens hashed in database
- [ ] Strict redirect URI matching (no wildcards)
- [ ] State parameter required and validated
- [ ] Client secrets hashed with Argon2
- [ ] Rate limiting on all endpoints

### Credential Broker
- [ ] Popup window (not iframe) for OAuth flows
- [ ] User can verify URL in address bar
- [ ] Explicit consent screen with scope details
- [ ] postMessage origin validation
- [ ] Nonce to prevent replay attacks
- [ ] Grant ID (not credential ID) exposed to clients

### Proxy API
- [ ] Path allowlists per scope
- [ ] SSRF prevention via strict URL validation
- [ ] Request body validation
- [ ] Response header sanitization
- [ ] Rate limiting per grant
- [ ] Audit logging for all requests

### General
- [ ] All endpoints HTTPS only
- [ ] HSTS enabled
- [ ] CSP headers on consent pages
- [ ] No sensitive data in logs
- [ ] Token rotation on security events

---

## Component 5: Agent Execution API

External apps need to run AutoGPT agents using the user's credentials. This requires careful API design to balance functionality with security.

### Security Principles for Agent Execution

1. **Capability-Based Discovery**: External apps see "what they can do", not raw credential lists
2. **Grant-Scoped Execution**: Agents can only use credentials the user granted to that specific app
3. **Execution Isolation**: One app's agent runs can't interfere with another's
4. **Cost Attribution**: Usage is tracked and attributed to the external app
5. **Audit Everything**: Full trail of what ran, with what credentials, triggered by whom

### Why NOT a Simple "List Credentials" Endpoint

Exposing raw credential lists creates problems:

| Issue | Risk |
|-------|------|
| **Information Leakage** | External app learns which services user has connected (even ones not granted to them) |
| **Credential ID Exposure** | Internal IDs could be used for enumeration attacks |
| **Scope Confusion** | App might try to use credentials beyond their granted scope |
| **Stale Data** | List could become outdated; better to validate at execution time |

**Instead**: Expose "capabilities" - what the app CAN do, not what EXISTS.

### API Design

#### GET /api/v1/capabilities

Returns what the external app can do with this user's account.

**Headers:**
```
Authorization: Bearer AUTOGPT_ACCESS_TOKEN
```

**Response:**
```json
{
    "user_id": "user_abc123",
    "client_id": "lvbl_xyz789",

    "grants": [
        {
            "grant_id": "grant_001",
            "provider": "google",
            "capabilities": [
                {
                    "scope": "gmail.readonly",
                    "description": "Read Gmail messages",
                    "available_actions": ["read_emails", "search_emails", "list_labels"]
                },
                {
                    "scope": "drive.readonly",
                    "description": "Read Google Drive files",
                    "available_actions": ["list_files", "read_file", "search_files"]
                }
            ],
            "granted_at": "2025-01-15T10:30:00Z",
            "expires_at": null
        },
        {
            "grant_id": "grant_002",
            "provider": "github",
            "capabilities": [
                {
                    "scope": "repos.read",
                    "description": "Read repository contents",
                    "available_actions": ["list_repos", "read_file", "list_branches"]
                }
            ],
            "granted_at": "2025-01-16T14:00:00Z",
            "expires_at": "2025-04-16T14:00:00Z"
        }
    ],

    "available_agents": [
        {
            "agent_id": "email_summarizer_v1",
            "name": "Email Summarizer",
            "description": "Summarizes recent emails",
            "required_capabilities": ["gmail.readonly"],
            "can_execute": true
        },
        {
            "agent_id": "code_reviewer_v1",
            "name": "Code Reviewer",
            "description": "Reviews pull requests",
            "required_capabilities": ["github.repos.read", "github.pulls.read"],
            "can_execute": false,
            "missing_capabilities": ["github.pulls.read"]
        }
    ],

    "limits": {
        "executions_remaining_today": 100,
        "executions_remaining_this_hour": 20,
        "max_concurrent_executions": 3
    }
}
```

**Key Design Decisions:**

1. **Grant IDs, not Credential IDs**: External apps reference grants (their access), not credentials (user's assets)
2. **Capabilities, not raw scopes**: Human-readable descriptions of what's possible
3. **Agent compatibility**: Shows which agents can run with current grants
4. **Rate limit visibility**: App knows its limits upfront

#### POST /api/v1/agents/{agent_id}/execute

Execute an agent using granted credentials.

**Headers:**
```
Authorization: Bearer AUTOGPT_ACCESS_TOKEN
Content-Type: application/json
```

**Request Body:**
```json
{
    "grants": ["grant_001"],
    "inputs": {
        "query": "Summarize emails from the last 24 hours about project deadlines"
    },
    "options": {
        "max_steps": 10,
        "timeout_seconds": 300,
        "webhook_url": "https://lovable.dev/webhooks/autogpt"
    }
}
```

**Validation Flow:**

```python
async def execute_agent(
    agent_id: str,
    request: AgentExecuteRequest,
    autogpt_token: str,
) -> AgentExecutionResponse:
    """
    Execute an agent with security validation at every step.
    """

    # Layer 1: Authenticate external app
    token_claims = verify_autogpt_token(autogpt_token)
    client_id = token_claims["client_id"]
    user_id = token_claims["sub"]

    # Layer 2: Verify client has 'agents:execute' scope
    if "agents:execute" not in token_claims["scope"]:
        raise HTTPException(403, "Token missing agents:execute scope")

    # Layer 3: Validate all referenced grants
    validated_grants = []
    for grant_id in request.grants:
        grant = await get_credential_grant(grant_id)

        if not grant:
            raise HTTPException(404, f"Grant {grant_id} not found")
        if grant.user_id != user_id:
            raise HTTPException(403, "Grant belongs to different user")
        if grant.client_id != client_id:
            raise HTTPException(403, "Grant belongs to different client")
        if grant.revoked_at:
            raise HTTPException(403, f"Grant {grant_id} has been revoked")
        if grant.expires_at and grant.expires_at < datetime.utcnow():
            raise HTTPException(403, f"Grant {grant_id} has expired")

        validated_grants.append(grant)

    # Layer 4: Load agent and verify requirements
    agent = await get_agent(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")

    # Check if agent is allowed for external execution
    if not agent.allow_external_execution:
        raise HTTPException(403, "Agent not available for external execution")

    # Layer 5: Verify grants satisfy agent requirements
    available_scopes = set()
    for grant in validated_grants:
        available_scopes.update(grant.granted_scopes)

    missing_scopes = set(agent.required_scopes) - available_scopes
    if missing_scopes:
        raise HTTPException(
            403,
            f"Missing required capabilities: {list(missing_scopes)}"
        )

    # Layer 6: Check rate limits
    if not await check_execution_rate_limit(client_id, user_id):
        raise HTTPException(429, "Execution rate limit exceeded")

    # Layer 7: Check concurrent execution limit
    active_executions = await count_active_executions(client_id, user_id)
    if active_executions >= MAX_CONCURRENT_EXECUTIONS:
        raise HTTPException(
            429,
            f"Maximum concurrent executions ({MAX_CONCURRENT_EXECUTIONS}) reached"
        )

    # Layer 8: Validate webhook URL if provided
    if request.options.webhook_url:
        if not is_valid_webhook_url(request.options.webhook_url, client_id):
            raise HTTPException(400, "Webhook URL not in registered domains")

    # Layer 9: Create execution record
    execution = await create_execution(
        agent_id=agent_id,
        user_id=user_id,
        client_id=client_id,
        grant_ids=[g.id for g in validated_grants],
        inputs=request.inputs,
        options=request.options,
    )

    # Layer 10: Audit log
    await audit_log.agent_execution_started(
        execution_id=execution.id,
        agent_id=agent_id,
        client_id=client_id,
        user_id=user_id,
        grant_ids=[g.id for g in validated_grants],
    )

    # Layer 11: Queue execution with credential resolver
    await execution_queue.enqueue(
        execution_id=execution.id,
        credential_resolver=GrantBasedCredentialResolver(
            grants=validated_grants,
            client_id=client_id,  # For audit attribution
        ),
    )

    return AgentExecutionResponse(
        execution_id=execution.id,
        status="queued",
        estimated_start_time=await estimate_start_time(),
    )
```

**Response (Async):**
```json
{
    "execution_id": "exec_abc123",
    "status": "queued",
    "estimated_start_time": "2025-01-15T10:30:05Z",
    "poll_url": "/api/v1/executions/exec_abc123",
    "webhook_registered": true
}
```

#### GET /api/v1/executions/{execution_id}

Poll execution status.

**Response (In Progress):**
```json
{
    "execution_id": "exec_abc123",
    "status": "running",
    "started_at": "2025-01-15T10:30:05Z",
    "current_step": 3,
    "max_steps": 10,
    "logs": [
        {"timestamp": "...", "message": "Fetching emails from Gmail..."},
        {"timestamp": "...", "message": "Found 47 emails in last 24 hours"},
        {"timestamp": "...", "message": "Analyzing email content..."}
    ]
}
```

**Response (Completed):**
```json
{
    "execution_id": "exec_abc123",
    "status": "completed",
    "started_at": "2025-01-15T10:30:05Z",
    "completed_at": "2025-01-15T10:31:45Z",
    "steps_executed": 7,
    "result": {
        "summary": "You have 3 emails about project deadlines...",
        "emails_processed": 47,
        "key_deadlines": [
            {"subject": "Q1 Report Due", "date": "2025-01-20"},
            {"subject": "Sprint Review", "date": "2025-01-18"}
        ]
    },
    "usage": {
        "llm_tokens": 15420,
        "api_calls": 12,
        "execution_time_seconds": 100
    }
}
```

#### POST /api/v1/executions/{execution_id}/cancel

Cancel a running execution.

```json
{
    "reason": "User requested cancellation"
}
```

### Credential Resolution During Execution

When an agent runs, it needs credentials. The `GrantBasedCredentialResolver` ensures proper isolation:

```python
class GrantBasedCredentialResolver:
    """
    Resolves credentials during agent execution,
    scoped to what the external app was granted.
    """

    def __init__(self, grants: list[CredentialGrant], client_id: str):
        self.grants = {g.credential_id: g for g in grants}
        self.client_id = client_id

    async def get_credential(
        self,
        provider: str,
        required_scopes: list[str]
    ) -> Optional[Credential]:
        """
        Get a credential for the given provider/scopes.
        Only returns credentials that were explicitly granted.
        """

        for credential_id, grant in self.grants.items():
            credential = await get_credential(credential_id)

            # Check provider matches
            if credential.provider != provider:
                continue

            # Check all required scopes are granted
            if not all(s in grant.granted_scopes for s in required_scopes):
                continue

            # Log credential access for audit
            await audit_log.credential_accessed(
                credential_id=credential_id,
                grant_id=grant.id,
                client_id=self.client_id,
                provider=provider,
                scopes=required_scopes,
            )

            return credential

        return None  # No matching grant found

    def get_available_providers(self) -> list[str]:
        """List providers available through grants (for agent introspection)."""
        providers = set()
        for credential_id in self.grants:
            credential = get_credential_sync(credential_id)
            providers.add(credential.provider)
        return list(providers)
```

### Webhook Notifications

External apps can receive execution updates via webhooks:

```python
class ExecutionWebhook:
    """Webhook payload for execution events"""

    event_type: Literal[
        "execution.started",
        "execution.step_completed",
        "execution.completed",
        "execution.failed",
        "execution.cancelled"
    ]

    execution_id: str
    timestamp: datetime

    # Event-specific data
    data: dict

    # Security
    signature: str  # HMAC-SHA256 of payload with client's webhook secret

# Webhook validation on client side
def verify_webhook(payload: bytes, signature: str, secret: str) -> bool:
    expected = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, expected)
```

### Database Schema Additions

```prisma
model AgentExecution {
    id              String   @id @default(uuid())

    agentId         String
    userId          String
    user            User     @relation(fields: [userId], references: [id])
    clientId        String
    client          OAuthClient @relation(fields: [clientId], references: [id])

    // Which grants were used
    grantIds        String[]

    // Execution state
    status          String   // queued, running, completed, failed, cancelled
    inputs          Json
    options         Json
    result          Json?
    error           String?

    // Timing
    queuedAt        DateTime @default(now())
    startedAt       DateTime?
    completedAt     DateTime?

    // Usage tracking
    stepsExecuted   Int      @default(0)
    llmTokensUsed   Int      @default(0)
    apiCallsMade    Int      @default(0)

    // Webhook
    webhookUrl      String?
    webhookSecret   String?  // Per-execution secret

    @@index([userId])
    @@index([clientId])
    @@index([status])
    @@index([queuedAt])
}
```

### Rate Limits for Agent Execution

| Limit | Value | Scope |
|-------|-------|-------|
| Executions per hour | 20 | per client + user |
| Executions per day | 100 | per client + user |
| Concurrent executions | 3 | per client + user |
| Max execution time | 10 min | per execution |
| Max steps per execution | 50 | per execution |

### Security Considerations for Agent Execution

| Risk | Mitigation |
|------|------------|
| **Credential scope creep** | Agent can ONLY access credentials through granted scopes |
| **Resource exhaustion** | Strict rate limits, execution timeouts, step limits |
| **Result data leakage** | Results only returned to the client that initiated execution |
| **Webhook SSRF** | Webhook URLs validated against registered domains |
| **Cost attribution bypass** | Usage tracked and attributed to client_id |
| **Execution tampering** | Execution IDs are UUIDs, ownership validated on every request |

### Example: Full External App Flow

```typescript
// External app (Lovable) complete integration

class AutoGPTClient {
    private accessToken: string;
    private baseUrl = 'https://autogpt.com/api/v1';

    // 1. Check what we can do
    async getCapabilities(): Promise<Capabilities> {
        const response = await fetch(`${this.baseUrl}/capabilities`, {
            headers: { 'Authorization': `Bearer ${this.accessToken}` }
        });
        return response.json();
    }

    // 2. Execute an agent
    async executeAgent(
        agentId: string,
        grantIds: string[],
        inputs: object
    ): Promise<ExecutionResponse> {
        const response = await fetch(`${this.baseUrl}/agents/${agentId}/execute`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.accessToken}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                grants: grantIds,
                inputs,
                options: {
                    max_steps: 10,
                    timeout_seconds: 300,
                    webhook_url: 'https://lovable.dev/webhooks/autogpt'
                }
            }),
        });
        return response.json();
    }

    // 3. Poll for results (or use webhooks)
    async pollExecution(executionId: string): Promise<ExecutionStatus> {
        const response = await fetch(
            `${this.baseUrl}/executions/${executionId}`,
            { headers: { 'Authorization': `Bearer ${this.accessToken}` } }
        );
        return response.json();
    }
}

// Usage
async function summarizeUserEmails(userId: string) {
    const client = new AutoGPTClient(getUserToken(userId));

    // Check capabilities
    const caps = await client.getCapabilities();

    // Find Gmail grant
    const gmailGrant = caps.grants.find(g =>
        g.provider === 'google' &&
        g.capabilities.some(c => c.scope === 'gmail.readonly')
    );

    if (!gmailGrant) {
        // Prompt user to connect Gmail
        return promptConnectGmail(userId);
    }

    // Execute email summarizer
    const execution = await client.executeAgent(
        'email_summarizer_v1',
        [gmailGrant.grant_id],
        { query: 'Summarize important emails from today' }
    );

    // Poll or wait for webhook
    let result;
    while (true) {
        const status = await client.pollExecution(execution.execution_id);
        if (status.status === 'completed') {
            result = status.result;
            break;
        }
        if (status.status === 'failed') {
            throw new Error(status.error);
        }
        await sleep(2000);
    }

    return result;
}
```

---

## Open Questions

1. **Client Approval**: Fully automated or manual review for new clients?
2. **Scope Granularity**: How fine-grained should integration scopes be?
3. **Grant Expiration**: Should grants expire? If so, what's the default?
4. **Offline Access**: Should proxy work when user is not actively using AutoGPT?
5. **Billing**: Meter proxy usage for monetization?
6. **Webhooks**: Notify clients when grants are revoked?

---

## References

- [OAuth 2.0 RFC 6749](https://tools.ietf.org/html/rfc6749)
- [PKCE RFC 7636](https://tools.ietf.org/html/rfc7636)
- [OAuth 2.0 Security Best Practices](https://tools.ietf.org/html/draft-ietf-oauth-security-topics)
- [OpenID Connect Core](https://openid.net/specs/openid-connect-core-1_0.html)
- [Existing OAuth Documentation](./oauth-integration-flow.md)
