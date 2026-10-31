# Client IP Addresses

## Overview

A few parts of the AutoGPT Platform count or record callers by their IP address:

- [Rate limits](rate-limits.md) that count per IP address: store search when signed out, and signing in and account emails.
- The sign-in history on an account's page under **Admin → Users**, which lists the IP addresses it signed up and signed in from.
- IP blocks, which an admin sets on that page to refuse sign-ups and sign-ins from an address.

This page explains where the backend and the frontend each get a caller's IP address, and what to set on a self-hosted install so it is the visitor's real one. On AutoGPT Cloud there is nothing to set.

## The backend

The backend's API server uses the address of the connection it receives. It takes the address from the `X-Forwarded-For` header instead only when the connection comes from an address listed in `FORWARDED_ALLOW_IPS`, a setting of the server (uvicorn) that defaults to `127.0.0.1,::1`.

If a reverse proxy sits in front of the backend, set `FORWARDED_ALLOW_IPS` on the backend to the proxy's address.

## The frontend

The frontend's server treats `X-Forwarded-For` as the visitor's address only when it runs on Vercel, or when `TRUST_X_FORWARDED_FOR=true` is set on the frontend. Set it only when the proxy in front of the frontend sets `X-Forwarded-For` to the visitor's address, replacing any value the request arrived with. Without it, the frontend has no visitor address to pass on.

The single-container image sets `TRUST_X_FORWARDED_FOR=true` itself, because its built-in nginx sets the header that way.

## From the frontend to the backend

Browsers reach the backend through the frontend's server, so the backend sees the frontend's address, not the visitor's. For store search, which the backend limits per IP address when the visitor is signed out, the frontend sends the visitor's address along in a signed token in the `X-Client-IP-Token` header, and the backend counts by that address.

- The token is signed with the frontend's sign-in key, which the backend already checks through `JWT_JWKS_URL`, and is valid for 60 seconds.
- When the frontend has no visitor address (see [the frontend](#the-frontend)), it sends no token. The backend then counts by the connection's address, so all signed-out store searches through one frontend server share one count.

## Signing in

Signing in uses the visitor's address in two ways:

- **The authentication API's [rate limits](rate-limits.md#the-authentication-api)** count per IP address. Better Auth, the frontend's sign-in library, keeps them.
- **The sign-in history and IP blocks** follow [the frontend](#the-frontend) rule above: they use the address only on Vercel or with `TRUST_X_FORWARDED_FOR=true`. Without it, no sign-in address is recorded and IP blocks have no address to match. Blocks on an email domain still apply.

Behind a proxy that overwrites `X-Forwarded-For` with the visitor's address, set `TRUST_X_FORWARDED_FOR=true` on the frontend.

The backend deletes the sign-in history of an account after 90 days, or after the number of days in `AUTH_IP_EVENT_RETENTION_DAYS` (1 to 3650).

## What to set

| Your install                                              | On the frontend                                                                                  | On the backend                                                                      |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| AutoGPT Cloud                                             | Nothing                                                                                          | Nothing                                                                             |
| Single-container image                                    | Nothing                                                                                          | Nothing                                                                             |
| Docker Compose, browsers connect to the frontend directly | Leave `TRUST_X_FORWARDED_FOR` unset                                                              | Nothing                                                                             |
| Behind a reverse proxy                                    | `TRUST_X_FORWARDED_FOR=true`, if the proxy sets `X-Forwarded-For` to the visitor's address alone | `FORWARDED_ALLOW_IPS=<proxy address>`, if the proxy is also in front of the backend |

On any install that serves the public, put the frontend behind a reverse proxy that sets `X-Forwarded-For` to the visitor's address alone, replacing any value the request arrived with.
