# Scheduling App — Calendly Alternative

A professional scheduling PWA that replaces Calendly by booking meetings directly into Google Calendar with automatic Google Meet links.

## Features

- **Monthly calendar** to select a date
- **30 or 60-minute slots** based on configurable working hours (09:00–18:00 Israel time)
- **Busy-slot filtering** — pulls existing events from your Google Calendar
- **Automatic Google Meet link** on every booking
- **Timezone-aware** — visitors see times in their local timezone
- **Booking form** with name, email, and meeting subject
- **Confirmation screen** with "Add to Calendar" (.ics) and "Copy Meet Link"
- **Telegram bot** — book meetings directly in Telegram
- **REST API** — integrate with any other app (Wix Velo, webhooks, etc.)
- **Wix iframe** — embed directly in your Wix site
- **Supabase** — booking records stored for admin review

---

## Architecture

```
Browser (React PWA)
    ├── GAS API  →  Google Apps Script  →  Google Calendar
    └── Supabase JS client  →  Supabase (booking log)

Express.js (Railway)
    ├── Serves React build (dist/)
    └── /api/* REST proxy (requires X-Api-Key)

Telegram Bot (Railway — separate process)
    └── Uses /api/* endpoints internally
```

---

## Quick Start (Local Development)

```bash
cd scheduling-app
cp .env.example .env          # fill in your values
npm install
npm run dev                   # http://localhost:5173
```

---

## Deployment on Railway

1. Push this repo to GitHub (`syncpartners1/scheduler-google`)
2. Create a new Railway project → **Deploy from GitHub repo**
3. Set environment variables (copy from `.env.example`)
4. Railway auto-builds and deploys on every push

The `railway.toml` configures the build and start commands automatically.

### Running the Telegram Bot

Railway supports multiple processes. The `Procfile` defines:
- `web` — Express server (main app)
- `bot` — Telegram bot

Both start automatically on Railway when you deploy.

---

## Google Apps Script Setup

See `gas/Code.gs` for full inline setup instructions.

**Short version:**
1. Go to https://script.google.com → New project
2. Paste `gas/Code.gs` content
3. Set `OWNER_CALENDAR_ID` to your Gmail address
4. Enable **Google Calendar API** (Extensions → Services)
5. Deploy → New deployment → **Web App** (Execute as: Me, Access: Anyone)
6. Copy the Web App URL → set as `VITE_GAS_URL` and `GAS_URL` in Railway

---

## Supabase Setup

Run `supabase/migrations/001_create_bookings.sql` in your Supabase SQL Editor.

The table stores booking records with Row Level Security:
- **Anon** can INSERT (the React app uses the anon key)
- **Authenticated** (admin) can SELECT

---

## REST API

All `/api/*` endpoints require `X-Api-Key: YOUR_API_KEY` header.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Liveness check |
| `GET` | `/api/slots?date=YYYY-MM-DD&tz=...&duration=30` | Available slots |
| `POST` | `/api/book` | Create a booking |

**POST /api/book body:**
```json
{
  "name": "Jane Smith",
  "email": "jane@example.com",
  "subject": "Product demo",
  "startISO": "2024-01-15T10:00:00.000Z",
  "duration": 30,
  "userTz": "America/New_York",
  "requestId": "optional-idempotency-key"
}
```

---

## Wix Iframe Integration

Add an **HTML Embed** element to your Wix page:

```html
<iframe
  src="https://YOUR-APP.railway.app?embed=true"
  width="100%"
  height="700px"
  style="border: none;"
  allow="clipboard-write"
></iframe>
```

The app detects the `embed=true` param and switches to a compact layout without header/footer.

On successful booking, it fires a `postMessage` to the parent Wix page:
```js
// In your Wix Velo code:
window.addEventListener('message', (e) => {
  if (e.data.type === 'BOOKING_SUCCESS') {
    console.log('Booking confirmed:', e.data.booking)
  }
})
```

---

## Telegram Bot

1. Create a bot via [@BotFather](https://t.me/BotFather) → `/newbot`
2. Copy the token → set `TELEGRAM_BOT_TOKEN` env var
3. Set `SERVER_URL` to your Railway app URL

Users chat with your bot:
- `/start` or `/book` — begin booking flow
- Pick date → pick duration → pick time slot → enter details → confirmation

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `VITE_GAS_URL` | Yes | GAS Web App URL (Vite build) |
| `GAS_URL` | Yes | GAS Web App URL (server-side) |
| `VITE_SUPABASE_URL` | Optional | Supabase project URL |
| `VITE_SUPABASE_ANON_KEY` | Optional | Supabase anon key |
| `VITE_OWNER_NAME` | Optional | Your name (shown in header) |
| `API_KEY` | Recommended | Protects `/api/*` endpoints |
| `TELEGRAM_BOT_TOKEN` | Optional | Enable Telegram bot |
| `SERVER_URL` | If using bot | Public Railway URL for bot→API calls |

---

## Constraints & Edge Cases

| Constraint | Where handled |
|-----------|---------------|
| 2-hour minimum notice | Front-end filter + GAS validation |
| 15-min buffer around events | GAS `getBusySlots` expands each event ±15 min |
| No double-booking | GAS conflict check before `createEvent` |
| Idempotent booking | GAS checks `requestId` in event description |
| Email validation | Front-end regex + GAS validation |
| `sendUpdates: 'all'` | GAS sends Google Calendar invite to both parties |
