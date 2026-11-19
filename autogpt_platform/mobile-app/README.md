# AutoGPT mobile app (proof of concept)

An Expo Router app for iOS and Android that talks to the same web app and backend as the site.

## What it does

- **Login:** email and password, existing accounts only.
- **Home:** laid out like the ChatGPT app. A pill at the top picks who you are talking to (Otto or one of your experts); the page stays empty, with a short list just above the composer: what needs you and what is running or next, then starters. The composer's main button dictates until there is something to send.
- **Chats:** as in the ChatGPT app, the menu slides in over three quarters of the screen and pushes the page aside, rounded and faded; tap what is left of the page to close it. It has search, your team, the Marketplace and "All chats" (every chat grouped by expert), then one row per expert with their latest chat, with a floating "New chat" pill and your avatar (settings, sign out) at its foot. In a chat you can send, stream, stop, load earlier messages, and pick up a reply that is still running or was dropped.
- **Chat feel:** a sent message moves to the top of the screen and the reply fills in below it at an even pace, without the view chasing it; an arrow brings you back to the end. The thread moves with the keyboard frame by frame, and dragging it down closes the keyboard. Long-press a message to copy or share it. The composer is the same card as on Home.
- **Opening fast:** the app saves the chat list, your experts and your last few chats on the phone, so they open at once and then catch up with the server. A chat also starts loading as your finger lands on its row.
- **Steps:** what the expert is doing shows as a live list and folds to one line when the reply lands.
- **Attach and dictate:** in a chat, attach photos and files; anywhere the composer is, dictate into the field. While you talk the mic button is an orb that moves with your voice; tap it to use what you said.
- **Connecting services:** when Otto or an expert needs a service, the chat shows a card with one row per service: sign in, add a key or login, reconnect, or allow an expert to use an account you already have. Once every row is connected, the chat carries on by itself after a sign-in from the card; a card whose accounts were already there, or one setting up a trigger, shows Continue instead.
- **Chat panel:** tap the name at the top of a chat; the panel slides in from the right the same way, pushing the chat aside; tap the chat or the back arrow to close it. The chat's header floats over the thread, which fades out under it. In an expert's chat, their profile with Activity (upcoming routines, recent runs, computer state) and Library (files) tabs; in a chat with Otto, the chat's files. Share the chat from its top bar.
- **Connections:** the expert panel's Connections tab lists the accounts the expert can use (Remove takes away their access, not the account) and what their workflows still need; a chat with Otto lists all your connections under its files. "Add connection" allows an account you have or connects a new service, an MCP server by address included.
- **Team:** the experts you have hired; tap one to chat. Create a new expert from the header, or hire one from the Marketplace.
- **Marketplace:** "Hire an expert" shows your team first, with an open seat for each area nobody covers, then the experts grouped by area (uncovered areas first) and a search by job or name; hire with the + on a row or from an expert's page (their area, first skills and all skills), and the new expert says hello in their first chat.
- **Settings:** the web `/settings` page in a WebView, reached from your avatar at the foot of the drawer.
- **Themes:** light and dark, following the phone's setting.

## What it does not do

Sign-up, onboarding, billing and paywalls, social login, push notifications, choosing a model, pausing a single routine, uploading or generating an expert's picture, deleting or downloading files, store builds, and CI.

## Run it

1. Run the frontend and backend as usual. Sign-in needs no extra setting.
2. In this folder:

   ```sh
   pnpm install
   cp .env.example .env.local
   npx expo start
   ```

3. Press `i` for the iOS simulator or `a` for the Android emulator, or scan the QR code with Expo Go.

`localhost` only works on the iOS simulator. On the Android emulator use `10.0.2.2` in `.env.local`; on a phone use your computer's address on the local network.

### How sign-in works

The app follows the auth service's contract for installed apps in [`docs/engineering/native-mobile/backend-handoff.md`](../../docs/engineering/native-mobile/backend-handoff.md), so the web app needs no change for it:

- It signs in at `/api/auth/sign-in/email` with the `X-AutoGPT-Native: 1` header and no cookie or Origin. The answer's `set-auth-token` header is the signed session credential, which the app keeps in the phone's key store, one per server.
- It sends that credential as a bearer to the auth service only: `/api/auth/token` swaps it for the short-lived token the backend takes, `/api/auth/get-session` checks at launch that the session still stands, and `/api/auth/sign-out` ends it.
- Dictation and Settings use web routes that read a cookie, so for those two the app sends the same credential as the session cookie.

A web app from before that contract answers a sign-in with a cookie only, and the app then says the server can't sign it in yet.

### Connecting services

Sign-ins open in the phone's own browser sheet (never a WebView, which Google refuses) and come back to `autogpt-app://integrations/oauth`. That needs a web app whose `/auth/integrations/oauth_callback` and `/auth/integrations/mcp_callback` routes send sign-ins started from the app back to it; with an older web app the sheet stays on its own page until you close it. Keys, logins and MCP tokens need no web change.

On iOS the system asks before every sign-in whether the app may use the web app's domain; that prompt can't be changed. On Android, Expo Go can't receive `autogpt-app://` links, so test sign-ins in a development build (`npx expo run:android`). A sign-in on a physical phone comes back to the web app's address, so `localhost` only works on a simulator.

## Environment

Set these in `.env.local` (see `.env.example`). Trailing slashes are dropped.

| Variable | Default | What it points at |
| --- | --- | --- |
| `EXPO_PUBLIC_FRONTEND_URL` | `http://localhost:3000` | The web app: sign-in (`/api/auth`) and the Settings page. |
| `EXPO_PUBLIC_API_URL` | `http://localhost:8006` | The backend: chats and experts. |

## Scripts

- `pnpm start`: start the dev server.
- `pnpm test`: unit tests (jest-expo).
- `pnpm types`: type check.
- `pnpm lint`: lint.

## Known limits

- Saved chats sit unencrypted in the app's own storage (SQLite) for up to a day, at most eight of them; signing out deletes them.
- Dictation uses the web app's `/api/transcribe` route, so the web server needs a transcription key; without one the app says dictation isn't available.
- The microphone and photo permission texts in `app.json` only apply in a development or store build, not in Expo Go.
- The drawer shows each expert's latest chat title, not their last message: the session list has no preview field.
- A "Needs you" action that is not a chat or the team page opens on the web.
- Attaching a file with a name already used in the same chat fails; the backend refuses it.
- File chips under a sent message are kept only until the chat reloads from the server.
- Settings shows the web page's own navigation, and is always light.
- Approval cards in a chat are not shown.
- A connect card that also asks for details, a service that signs in with a device code (such as Codex) or through Ayrshare, and a connect that hits a paywall all send you to the web to finish.
- Images in Otto's replies show as their alt text only.
- Only `http` and `https` links open.
- Plain HTTP only works in development; release builds need HTTPS.
- The Settings WebView sets the session cookie with `document.cookie`, so inside the WebView it is not HttpOnly.
