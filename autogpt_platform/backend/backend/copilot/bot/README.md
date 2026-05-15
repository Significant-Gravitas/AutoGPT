# CoPilot Bot

Multi-platform chat bot that bridges AutoPilot to Discord and Slack (with Telegram, Teams, WhatsApp on the roadmap).

## Running

```bash
# As a standalone service
poetry run copilot-bot

# Or auto-start alongside the rest of the platform
poetry run app   # starts the bot too if AUTOPILOT_BOT_DISCORD_TOKEN is set
```

## Required environment variables

See `backend/.env.default` for the full list with documentation. Minimum setup:

| Variable | Purpose |
|----------|---------|
| `AUTOPILOT_BOT_DISCORD_TOKEN` | Discord bot token — enables the Discord adapter |
| `AUTOPILOT_BOT_SLACK_TOKEN` | Slack bot OAuth token (`xoxb-…`) — pair with the signing secret to enable Slack |
| `AUTOPILOT_BOT_SLACK_SIGNING_SECRET` | Slack signing secret for verifying inbound Events API + slash command requests |
| `FRONTEND_BASE_URL` | Frontend base URL for link confirmation pages (shared with the rest of the backend) |
| `REDIS_HOST` / `REDIS_PORT` | Session + thread subscription state + copilot stream subscription (inherited from the shared backend config) |
| `PLATFORMLINKINGMANAGER_HOST` | DNS name of the `PlatformLinkingManager` service pod (cluster-internal RPC) |

## Architecture

```
bot/
├── app.py              # CoPilotChatBridge(AppService) — runs socket adapters (Discord)
├── webhook_routes.py   # register_webhook_adapters(app) — mounts webhook adapters onto the main API
├── config.py           # Shared (platform-agnostic) config
├── handler.py          # Core logic: routing, linking, batched streaming
├── bot_backend.py      # Thin facade over PlatformLinkingManagerClient + stream_registry
├── text.py             # Text splitting + batch formatting
├── threads.py          # Redis-backed thread subscription tracking
└── adapters/
    ├── base.py         # PlatformAdapter + SocketAdapter / WebhookAdapter, MessageContext
    ├── discord/
    │   ├── adapter.py  # Gateway connection, events, sends, thread creation
    │   ├── commands.py # Slash commands (/setup, /help, /unlink)
    │   └── config.py   # Discord token + platform limits
    └── slack/
        ├── adapter.py       # Events API + slash command routes, sends, thread routing
        ├── app-manifest.yaml# Slack app manifest for reproducible app creation
        ├── commands.py      # Slash commands (/setup, /help, /unlink)
        ├── config.py        # Slack token + signing secret + platform limits
        ├── signing.py       # HMAC-SHA256 request signature verification
        └── text.py          # Markdown → mrkdwn + mention substitution
```

**Connector types:** adapters extend one of two base classes — `SocketAdapter`
(holds a long-lived per-token connection; Discord today) or `WebhookAdapter`
(receives inbound HTTPS POSTs; stateless, mounted onto the main backend API
via `webhook_routes.register_webhook_adapters` — Slack today).

**Locality rule:** anything platform-specific lives under `adapters/<platform>/`.
The two factory functions (`app.py::_build_socket_adapters` and
`webhook_routes.py::_build_webhook_adapters`) are the only files that name
specific platforms — they decide which adapters to instantiate based on
configured credentials.

## How messaging works

1. User mentions the bot in a channel
2. Adapter's `on_message` handler fires, constructs a `MessageContext`, passes
   it to the shared `MessageHandler`
3. Handler:
   - Checks if the user/server is linked (via `bot_backend`)
   - If not linked → sends a "Link Account" button prompt
   - If linked → creates a thread (for channels) or uses the existing thread/DM
   - Marks the thread as subscribed in Redis (7-day TTL)
   - Streams the AutoPilot response back, chunked at the adapter's
     `chunk_flush_at` boundary
4. Messages that arrive while a stream is running get batched and sent as a
   single follow-up turn once the current stream ends

## Adding a new platform

Adapters extend one of two base classes (see **Connector types** above):

- **Socket-owning** — subclass `SocketAdapter`, implement `start`/`stop`, and
  register it in `app.py::_build_socket_adapters`.
- **Webhook-based** — subclass `WebhookAdapter` and implement `register_routes`
  (owning the request signature verification), then register it in
  `webhook_routes.py::_build_webhook_adapters`.

Then:

1. Create `adapters/<platform>/` with `adapter.py`, `config.py`, and
   `commands.py` (if the platform has commands)
2. `adapter.py` implements all the `PlatformAdapter` abstract methods —
   `send_message`, `send_link`, `create_thread`, `max_message_length`,
   `chunk_flush_at`, etc.
3. `config.py` declares the platform's env vars and any platform-specific
   numbers (message limits, token name, etc.)
4. Register the adapter in its factory:
   ```python
   if <platform>_config.get_bot_token():
       adapters.append(<Platform>Adapter(api))
   ```

The core handler, text utilities, thread tracking, and platform API all stay
untouched.
