# Sign in with Apple for native and browser clients

This follow-up uses the session foundation in [#15384](https://github.com/Significant-Gravitas/AutoGPT/pull/15384). Better Auth runs the server-side exchange under `autogpt_platform/frontend/src/lib/auth/`; the Python REST and WebSocket servers continue accepting the same session-bound access JWTs.

## Native configuration

1. Enable Sign in with Apple for the app's App ID in the Apple Developer account and its signing entitlements.
2. Set `AUTH_APPLE_APP_BUNDLE_IDENTIFIER` on the auth service to that exact bundle identifier. An empty value keeps the native bundle audience disabled. This is a public identifier, not a secret.
3. Deploy the auth service with the native session foundation available on the backend.

The native identity-token exchange needs Apple's public signing keys, fetched from `https://appleid.apple.com/auth/keys`; it needs no Apple private key, client-secret JWT, Service ID or browser callback registration. With native-only configuration, the configured bundle identifier is the only accepted audience.

## Browser configuration

1. Create an Apple Service ID and enable Sign in with Apple. Associate it with the same primary App ID as the native application so Apple identifies a person consistently across both clients.
2. Register the auth service's domain and exact HTTPS return URL: `https://<auth-host>/api/auth/callback/apple`. Apple does not accept localhost or non-HTTPS return URLs. Set `BETTER_AUTH_URL` to that public auth origin.
3. Create a Sign in with Apple key for the primary App ID. Supply these server-side values through the deployment's secret configuration:

   | Setting | Value |
   | --- | --- |
   | `AUTH_APPLE_CLIENT_ID` | Apple Service ID used by browser OAuth |
   | `AUTH_APPLE_TEAM_ID` | Apple Developer Team ID |
   | `AUTH_APPLE_KEY_ID` | Signing key's Key ID |
   | `AUTH_APPLE_PRIVATE_KEY` | Downloaded PKCS#8 `.p8` private key |

   The key accepts real newlines or escaped `\n` sequences. Keep it server-side; never put it in a `NEXT_PUBLIC_*` variable or commit it. This is an Apple-issued credential, not a locally generated application secret.
4. Deploy the auth service. Rotate the Apple key through the same deployment configuration and restart the service when changing credentials.

All four browser settings are required. When native login is configured, incomplete browser configuration returns `APPLE_WEB_LOGIN_NOT_CONFIGURED` for redirect requests while native login remains available. Browser-only configuration is supported without a native bundle identifier. Without a native bundle identifier or complete browser settings, the Apple provider is disabled and returns the standard `PROVIDER_NOT_FOUND` response.

The auth service signs a fresh ES256 client-secret JWT with a five-minute lifetime for each authorization, code exchange and provider-token refresh. Better Auth caches provider options at initialization, so the Apple plugin refreshes that credential when each operation runs, including after a long consent flow. The private key and client-secret JWT never go to the browser. Browser callback identities must have the configured Service ID audience; direct ID-token exchanges accept the configured native and Service ID audiences.

Apple signing keys are parsed when a browser OAuth operation runs, never during shared auth initialization. A malformed key or signing failure returns HTTP 503 `APPLE_SIGN_IN_UNAVAILABLE` when starting browser Apple login and emits a warning without credential values. Email login and configured native Apple login remain available. Correct the key configuration and restart the auth service before retrying browser Apple login.

## Browser exchange

The existing browser login endpoint accepts Apple:

```http
POST /api/auth/login/with-provider
Content-Type: application/json
```

```json
{
  "provider": "apple",
  "redirectTo": "/auth/callback"
}
```

Navigate to the returned `url`. The Better Auth client can also call `signIn.social({ provider: "apple", callbackURL: "/auth/callback" })`, or POST the same provider and callback URL to `/api/auth/sign-in/social`.

Apple posts its result to `/api/auth/callback/apple`. Better Auth redirects that form POST to a same-site GET, where the browser's OAuth state cookie is available, validates and consumes the state, and exchanges the authorization code server-side. The callback then creates the normal HttpOnly session cookies and redirects to the requested local destination. Only this Apple callback POST trusts Apple's origin; other auth endpoints keep their existing origin checks. Browser sessions receive no native bearer header. Missing or mismatched state, replayed callbacks, declined consent and invalid identity tokens do not create sessions.

This backend change exposes the provider through the existing login API; it does not add login buttons to the web or native UI.

## Native exchange

For each Apple authorization, generate a fresh cryptographically random nonce. Set `ASAuthorizationAppleIDRequest.nonce` to its SHA-256 hexadecimal digest and request the email and full-name scopes. Send the resulting identity token and original nonce to the selected server:

```http
POST /api/auth/sign-in/social
Content-Type: application/json
X-AutoGPT-Native: 1
```

```json
{
  "provider": "apple",
  "idToken": {
    "token": "<Apple identity token>",
    "nonce": "<original random nonce>",
    "user": {
      "name": {
        "firstName": "<first name, when supplied>",
        "lastName": "<last name, when supplied>"
      }
    }
  }
}
```

The `user` object is optional: Apple may supply the name only on initial consent. Identity and email come from the verified token, never a client-supplied email. Private relay addresses are supported. If a returning token omits email, the server uses only the user already bound to that verified Apple subject; an unknown subject without an email is rejected.

Send no Cookie, Origin or browser Fetch Metadata headers. Store the signed `set-auth-token` response header in Keychain, partitioned by server and account; the unsigned `token` in the JSON body is not a usable bearer credential. Exchange the signed session bearer at `GET /api/auth/token` and send the returned access JWT to the backend, as described in the [client contract](backend-handoff.md#contract-the-client-can-use). Renewal, platform bans, session expiry and logout use the existing endpoints and session-authority checks.

The server requires the nonce and uses Better Auth's Apple verifier for the RS256 signature, Apple issuer, configured audience, required claims, expiry and token age. Both a directly bound nonce and Apple's SHA-256 form are supported. Missing nonce returns HTTP 400; an invalid identity token returns HTTP 401. Apple signing-key retrieval failures also fail closed with HTTP 401 under the library's verifier; restart sign-in when connectivity recovers. A signed-in browser retains HttpOnly cookies and receives no native bearer header.

## Validation

`apple-sign-in.test.ts` runs signed Apple identity tokens through the real production auth configuration and Better Auth HTTP handler, including signed session export, access-JWT verification and renewal, repeat login, private relay email, browser cookie isolation and logout. `apple-sign-in-rejections.test.ts` covers signature/claim/nonce rejection, missing configuration, unavailable signing keys, account ownership, unverified-email linking, bans and signup policy.

The `apple-browser-*.test.ts` suites exercise authorization through Apple's form callback to a cookie session, logout, shared native/browser identity, fresh client secrets after delayed consent, browser-only and partial configuration, callback replay, origin/state checks, callback token validation and signup policy. Tests generate signing keys in memory and verify outgoing client-secret JWTs. Only external Apple responses, database storage, DNS and platform provisioning are replaced; JWT verification and session issuance are real. Origin and CSRF checks are explicitly enabled, including under the test environment.

Run them with the existing auth regressions from `autogpt_platform/frontend`:

```sh
pnpm test:unit src/lib/auth/__tests__
```

Live acceptance still requires configured Apple identifiers and credentials, a registered HTTPS domain and a signed native build: authorize with a real account in both clients, test Hide My Email and repeat consent, confirm the same account is selected, renew an access JWT against the deployed backend, then verify logout refuses further authenticated requests. No real Apple credentials or live Apple authorization are exercised by these automated tests.

References: [Better Auth Apple provider](https://better-auth.com/docs/authentication/apple), [Apple user verification](https://developer.apple.com/documentation/signinwithapple/verifying-a-user).
