# AutoGPT mobile app (proof of concept)

An Expo Router app for iOS and Android that talks to the same web app and backend as the site.

## What it does

- **Login:** email and password, existing accounts only.
- **Home:** a greeting, prompt chips and a composer. The first message starts a chat with Otto.
- **Chats:** the list in the drawer; open one, send, stream, stop, and pick up a reply that is still running.
- **Team:** the experts you have hired. Tap one to start a chat with that expert.
- **Settings:** the web `/settings` page in a WebView.

## What it does not do

Sign-up, onboarding, billing and paywalls, social login, connecting integrations, file attachments, voice, push notifications, choosing a model, hiring or editing experts, store builds, and CI.

## Run it

1. In `autogpt_platform/frontend/.env` set `AUTH_MOBILE_APP_SCHEME=autogpt-app`, then run the frontend and backend as usual.
2. In this folder:

   ```sh
   pnpm install
   cp .env.example .env.local
   npx expo start
   ```

3. Press `i` for the iOS simulator or `a` for the Android emulator, or scan the QR code with Expo Go.

`localhost` only works on the iOS simulator. On the Android emulator use `10.0.2.2` in `.env.local`; on a phone use your computer's address on the local network.

### How sign-in works on the server

With `AUTH_MOBILE_APP_SCHEME=autogpt-app`, the web app's auth accepts a request whose `expo-origin` header is exactly `autogpt-app://` (and, in development only, any `exp://` origin from Expo Go) as if it came from the site itself. The app scheme is never trusted as a redirect or callback URL, so the app sends no `callbackURL`. Without the variable, auth works as before.

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

- Settings shows the web page's own navigation.
- Credential and approval cards in a chat are not shown.
- Images in Otto's replies show as their alt text only.
- Only `http` and `https` links open.
- No push notifications.
- Plain HTTP only works in development; release builds need HTTPS.
- The Settings WebView sets the session cookie with `document.cookie`, so inside the WebView it is not HttpOnly.
- Nothing on this branch has been run on a device or simulator yet.
