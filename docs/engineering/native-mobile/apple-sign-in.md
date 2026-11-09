# Native Sign in with Apple

This follow-up uses the session foundation in [#15384](https://github.com/Significant-Gravitas/AutoGPT/pull/15384). Better Auth runs the server-side exchange under `autogpt_platform/frontend/src/lib/auth/`; the Python REST and WebSocket servers continue accepting the same session-bound access JWTs.

## Enable the provider

1. Enable Sign in with Apple for the app's App ID in the Apple Developer account and its signing entitlements.
2. Set `AUTH_APPLE_APP_BUNDLE_IDENTIFIER` on the auth service to that exact bundle identifier. An empty value keeps Apple sign-in disabled. This is a public identifier, not a secret.
3. Deploy the auth service with the native session foundation available on the backend.

The native identity-token exchange needs Apple's public signing keys, fetched from `https://appleid.apple.com/auth/keys`; it needs no Apple private key, client-secret JWT, Service ID or browser callback registration. The configured App ID is the only accepted audience. Browser OAuth authorization-code redirects are outside this change and return `APPLE_ID_TOKEN_REQUIRED` rather than starting an unconfigured redirect flow.

## Client exchange

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

`apple-sign-in.test.ts` runs signed Apple identity tokens through the real production auth configuration and Better Auth HTTP handler, including signed session export, access-JWT verification and renewal, repeat login, private relay email, browser cookie isolation and logout. `apple-sign-in-rejections.test.ts` covers signature/claim/nonce rejection, missing configuration, unavailable signing keys, account ownership, unverified-email linking, bans and signup policy. Only external Apple responses, database storage, DNS and platform provisioning are replaced in these tests; JWT verification and session issuance are real.

Run them with the existing auth regressions from `autogpt_platform/frontend`:

```sh
pnpm test:unit src/lib/auth/__tests__
```

Live acceptance still requires the app's registered App ID and a signed Apple build: authorize with a real account, test Hide My Email and repeat consent, renew an access JWT against the deployed backend, then verify logout refuses further authenticated requests. This PR supplies the server contract; it does not add a native button or claim live Apple certification.

References: [Better Auth Apple provider](https://better-auth.com/docs/authentication/apple), [Apple user verification](https://developer.apple.com/documentation/signinwithapple/verifying-a-user).
