# Stripe Link CLI Block — Auth Exploration

## What is Stripe Link CLI?

[`@stripe/link-cli`](https://github.com/stripe/link-cli) lets AI agents get secure,
one-time-use payment credentials from a user's **Link wallet** (Stripe's consumer
payment product). The core operations are:

| Operation | Description |
|-----------|-------------|
| `auth login` | Authenticate the agent with a Link account |
| `payment-methods list` | List cards/bank accounts in the wallet |
| `spend-request create` | Request a one-time virtual card credential |
| `spend-request retrieve` | Get card details once user approves |
| `mpp pay` | Execute payment via Machine Payments Protocol (SPT) |

## Link CLI's Auth Model

Link CLI uses **OAuth 2.0 Device Code Grant** ([RFC 8628](https://tools.ietf.org/html/rfc8628)):

```
┌─────────┐                        ┌──────────────────┐                   ┌──────────┐
│ AutoGPT │                        │ login.link.com   │                   │ User's   │
│ Backend │                        │ (Auth Server)    │                   │ Browser  │
└────┬────┘                        └────────┬─────────┘                   └────┬─────┘
     │                                      │                                  │
     │ POST /device/code                    │                                  │
     │  client_id=lwlpk_U7Qy7ThG69STZk     │                                  │
     │  scope=userinfo:read                 │                                  │
     │        payment_methods.agentic       │                                  │
     │─────────────────────────────────────>│                                  │
     │                                      │                                  │
     │  { device_code, user_code,           │                                  │
     │    verification_uri,                 │                                  │
     │    verification_uri_complete }       │                                  │
     │<─────────────────────────────────────│                                  │
     │                                      │                                  │
     │  Show verification_uri to user ──────┼──────────────────────────────────>│
     │                                      │                                  │
     │                                      │ User visits URL, logs in,        │
     │                                      │ enters user_code phrase          │
     │                                      │<─────────────────────────────────│
     │                                      │                                  │
     │ POST /device/token (poll)            │                                  │
     │  grant_type=device_code              │                                  │
     │  device_code=...                     │                                  │
     │─────────────────────────────────────>│                                  │
     │                                      │                                  │
     │  { access_token, refresh_token,      │                                  │
     │    expires_in, token_type }          │                                  │
     │<─────────────────────────────────────│                                  │
     │                                      │                                  │
```

**Key details:**
- **Client ID**: Hardcoded public `lwlpk_U7Qy7ThG69STZk` (no client_secret needed)
- **Scopes**: `userinfo:read payment_methods.agentic`
- **Token refresh**: Standard `refresh_token` grant at same `/device/token` endpoint
- **API calls**: Bearer token in `Authorization` header to `api.link.com`
- **User approval**: Push notification or email via Link app, with code-phrase confirmation

## AutoGPT's Current OAuth Model

AutoGPT uses the **Authorization Code Grant** flow:

1. Backend generates `login_url` → frontend redirects user to provider
2. User authorizes → provider redirects back with `code`
3. Backend exchanges `code` for tokens via `POST /{provider}/callback`
4. Tokens stored as `OAuth2Credentials` (access_token + refresh_token)

The `BaseOAuthHandler` interface:
```python
class BaseOAuthHandler(ABC):
    def __init__(self, client_id, client_secret, redirect_uri): ...
    def get_login_url(self, scopes, state, code_challenge) -> str: ...
    async def exchange_code_for_tokens(self, code, scopes, code_verifier) -> OAuth2Credentials: ...
    async def _refresh_tokens(self, credentials) -> OAuth2Credentials: ...
    async def revoke_tokens(self, credentials) -> bool: ...
```

## The Mismatch

| Aspect | Authorization Code (current) | Device Code (Link CLI) |
|--------|------------------------------|----------------------|
| **Initiation** | Redirect user to login URL | Show verification URL + code phrase |
| **Token acquisition** | One-shot callback with `code` | Poll `/device/token` until approved |
| **Client secret** | Required | Not used (public client) |
| **Redirect URI** | Required | Not used |
| **User interaction** | Same browser (redirect) | Separate device (phone/other browser) |

## Implementation Options

### Option A: Adapt `BaseOAuthHandler` (Minimal Backend Changes)

Map the device code flow onto the existing handler interface:

- `get_login_url()` → call `/device/code`, return `verification_uri_complete`
  - Store `device_code` in the OAuth state token
- `exchange_code_for_tokens()` → poll `/device/token` with the stored `device_code`
  - The `code` parameter is repurposed as the device_code
- `_refresh_tokens()` → standard refresh_token grant ✅
- `revoke_tokens()` → call `/device/revoke` ✅

**Pros:** Minimal backend changes, reuses existing OAuth infrastructure
**Cons:**
- Frontend redirect UX doesn't match — need to show "visit URL" instead of redirect
- Polling doesn't fit the one-shot callback model — `exchange_code_for_tokens` would
  need to block/poll (potentially for minutes)
- The frontend currently opens a popup/redirect; it would need a new "device auth" UI mode

### Option B: API Key Credential (Simplest)

Let users paste a pre-obtained access token as an API key:

```python
StripeLinkCredentials = APIKeyCredentials
StripeLinkCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.STRIPE_LINK], Literal["api_key"]
]
```

**Pros:** Zero infrastructure changes, works today
**Cons:**
- Terrible UX — user must run `link-cli auth login` externally, copy token
- No auto-refresh — tokens expire, user must re-authenticate manually
- Defeats the purpose of integrated credential management

### Option C: New Device Code OAuth Flow (Recommended)

Add a **device code flow variant** to the integrations system:

1. **New backend endpoint**: `POST /integrations/{provider}/device-auth`
   - Calls Link's `/device/code`
   - Returns `{ verification_url, user_code, poll_token }` to frontend
2. **New backend endpoint**: `GET /integrations/{provider}/device-auth/poll`
   - Polls Link's `/device/token` on demand
   - Returns `{ status: "pending" | "approved" | "denied" }`
   - On "approved", stores `OAuth2Credentials` and returns credential metadata
3. **Frontend UI**: Show verification URL + code phrase, poll status
4. **OAuth handler**: New `BaseDeviceAuthHandler` base class

```python
class BaseDeviceAuthHandler(ABC):
    """Handler for OAuth 2.0 Device Code Grant flows"""
    PROVIDER_NAME: ClassVar[ProviderName | str]

    @abstractmethod
    async def initiate_device_auth(self, scopes: list[str]) -> DeviceAuthState: ...

    @abstractmethod
    async def poll_device_auth(self, device_code: str) -> OAuth2Credentials | None: ...

    @abstractmethod
    async def _refresh_tokens(self, credentials: OAuth2Credentials) -> OAuth2Credentials: ...

    @abstractmethod
    async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool: ...
```

**Pros:**
- Clean separation of concerns
- Reusable for other device-code providers (smart TVs, CLI tools, IoT)
- Good UX — user sees clear instructions, approval via Link app
- Auto-refresh works via standard OAuth2Credentials

**Cons:**
- Requires new API endpoints and frontend UI components
- More implementation effort upfront

### Option D: Backend Polling with SSE/WebSocket (Best UX, Most Complex)

Like Option C but the backend polls automatically and pushes status to frontend via SSE:

- Backend initiates device auth and starts polling in a background task
- Frontend connects via SSE or WebSocket for real-time status updates
- When approved, credentials are stored and frontend is notified instantly

**Pros:** Best UX (no frontend polling), instant notification
**Cons:** Significant complexity, SSE/WebSocket infrastructure needed

## Recommendation

**Start with Option C** (New Device Code OAuth Flow). It's the cleanest architecture
that properly handles the device code flow without hacking it into the authorization
code flow. The endpoints and handler abstraction are also reusable for future providers.

**Fallback to Option A** if we want a quick proof of concept — the main challenge is
frontend UX, but the backend adaptation is relatively straightforward.

## What Blocks Would Look Like

Regardless of auth approach, the block surface area would be:

| Block | Description | Auth Scope |
|-------|-------------|------------|
| `StripeLinkListPaymentMethodsBlock` | List cards/bank accounts | `payment_methods.agentic` |
| `StripeLinkCreateCardSpendRequestBlock` | Create a virtual-card spend request (self-hosted only) | `payment_methods.agentic` |
| `StripeLinkGetSpendRequestStatusBlock` | Poll a spend request for approval | `payment_methods.agentic` |
| `StripeLinkRetrieveCardBlock` | Get the virtual card once approved (self-hosted only) | `payment_methods.agentic` |
| `StripeLinkRequestApprovalBlock` | Request user approval for spend | `payment_methods.agentic` |
| `StripeLinkMPPPayBlock` | Execute MPP payment | `payment_methods.agentic` |

All blocks would use the same credential type:
```python
StripeLinkCredentials = OAuth2Credentials
StripeLinkCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.STRIPE_LINK], Literal["oauth2"]
]
```

See `_auth.py` and `spend_request.py` in this directory for skeleton implementations.
