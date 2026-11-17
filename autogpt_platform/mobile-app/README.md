# AutoGPT mobile app (proof of concept)

An Expo Router app for iOS and Android that talks to the same web app and backend as the site.

## What it does

- **Login:** email and password, existing accounts only.
- **Home:** laid out like the ChatGPT app. A pill at the top picks who you are talking to (Otto or one of your experts); the page stays empty, with a short list just above the composer: what needs you and what is running or next, then starters. The composer's main button dictates until there is something to send.
- **Chats:** the drawer covers the whole screen, pushing the page aside, and lists one row per expert with their latest chat and has search; "All chats" lists every chat grouped by expert. In a chat you can send, stream, stop, load earlier messages, and pick up a reply that is still running or was dropped.
- **Chat feel:** a sent message moves to the top of the screen and the reply fills in below it at an even pace, without the view chasing it; an arrow brings you back to the end. The thread moves with the keyboard frame by frame, and dragging it down closes the keyboard. Long-press a message to copy or share it. The composer is the same card as on Home.
- **Opening fast:** the app saves the chat list, your experts and your last few chats on the phone, so they open at once and then catch up with the server. A chat also starts loading as your finger lands on its row.
- **Steps:** what the expert is doing shows as a live list and folds to one line when the reply lands.
- **Attach and dictate:** in a chat, attach photos and files; anywhere the composer is, dictate into the field.
- **Chat panel:** in an expert's chat, their profile with Activity (upcoming routines, recent runs, computer state) and Library (files) tabs; in a chat with Otto, the chat's files. Share the chat from its top bar.
- **Team:** the experts you have hired; tap one to chat. Create a new expert from the header, or hire one from the Marketplace.
- **Marketplace:** every expert you can hire, with search and areas. Hire from the list or from an expert's page, which shows what they do and their skills; the new expert says hello in their first chat.
- **Settings:** the web `/settings` page in a WebView, reached from the user row at the foot of the drawer.
- **Themes:** light and dark, following the phone's setting.

## What it does not do

Sign-up, onboarding, billing and paywalls, social login, connecting integrations, push notifications, choosing a model, pausing a single routine, uploading or generating an expert's picture, deleting or downloading files, store builds, and CI.

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
- Credential and approval cards in a chat are not shown.
- Images in Otto's replies show as their alt text only.
- Only `http` and `https` links open.
- Plain HTTP only works in development; release builds need HTTPS.
- The Settings WebView sets the session cookie with `document.cookie`, so inside the WebView it is not HttpOnly.
