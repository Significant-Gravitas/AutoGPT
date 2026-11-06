# Client IP Addresses

## Overview

A few parts of the AutoGPT Platform count or record callers by their IP address:

- [Rate limits](rate-limits.md) that count per IP address: store search when signed out, signing in and account emails, and calls to the External API without a key.
- The sign-in history on an account's page under **Admin → Users**, which lists the IP addresses it signed up and signed in from.
- IP blocks, which an admin sets on that page to refuse sign-ups and sign-ins from an address.

This page explains where the backend and the frontend each get a caller's IP address, and what to set on a self-hosted install so it is the visitor's real one. On AutoGPT Cloud there is nothing to set.

## The backend

The backend's API server uses the address of the connection it receives. It takes the address from the `X-Forwarded-For` header instead only when the connection comes from an address listed in `FORWARDED_ALLOW_IPS`, a setting of the server (uvicorn) that defaults to `127.0.0.1,::1`.

If a reverse proxy sits in front of the backend, set `FORWARDED_ALLOW_IPS` on the backend to the address it connects from — not necessarily the proxy's own address. A proxy on the host that reaches the backend through its published port (`8006` by default under Docker Compose) connects from the Docker network's gateway, not from `127.0.0.1`. Find it with `docker network inspect app-network --format '{{(index .IPAM.Config 0).Gateway}}'` (it varies per install; Docker assigns the subnet). A proxy running in its own container on the same network connects with its own container address instead.

The setting takes a comma-separated list of addresses or CIDR ranges, so `FORWARDED_ALLOW_IPS=172.18.0.0/16` covers a Docker subnet whose gateway address Docker may reassign.

Once the backend trusts that address, let nothing else reach it there: the backend reads `X-Forwarded-For` from any connection that comes from a trusted address. Under Docker Compose, publish the backend's port on the host's loopback only (`ports: ["127.0.0.1:8006:8006"]`), so only a proxy on the host can reach it, or drop the mapping and run the proxy in a container on the same network.

### Limits on the External API

The [External API](rate-limits.md#the-external-api)'s per-IP limits don't use the connection address. They read `X-Forwarded-For` and take the entry `TRUSTED_PROXY_COUNT` places from the right, because everything further left is whatever the caller sent. The default, `3`, is AutoGPT Cloud's own chain: Cloudflare appends the caller's address, and Google's load balancer appends two more.

Set `TRUSTED_PROXY_COUNT` on the backend to the number of entries your own proxies add — `1` for a single nginx or Caddy — or `0` to ignore the header and count by the connection address. With fewer entries than the setting the limit counts by the connection address, so every caller shares one count; with more, it counts by an entry the caller chose. The [single-container image](single-container.md) sets `TRUSTED_PROXY_COUNT=1` itself.

## The frontend

The frontend's server treats `X-Forwarded-For` as the visitor's address only when it runs on Vercel, or when `TRUST_X_FORWARDED_FOR=true` is set on the frontend. Set it only when the proxy in front of the frontend sets `X-Forwarded-For` to the visitor's address, replacing any value the request arrived with. Without it, the frontend has no visitor address to pass on.

The single-container image sets `TRUST_X_FORWARDED_FOR=true` itself, because its built-in nginx sets the header that way.

## From the frontend to the backend

Browsers reach the backend through the frontend's server, so the backend sees the frontend's address, not the visitor's. For store search, which the backend limits per IP address when the visitor is signed out, the frontend sends the visitor's address along in a signed token in the `X-Client-IP-Token` header, and the backend counts by that address.

- The token is signed with the frontend's sign-in key, which the backend already checks through `JWT_JWKS_URL`, and is valid for 60 seconds.
- When the frontend has no visitor address (see [the frontend](#the-frontend)), it sends no token. The backend then counts by the connection's address, so all signed-out store searches through one frontend server share one count.

## Signing in

Everything on the sign-in path reads the visitor's address through Better Auth, the frontend's sign-in library (1.7.12), which parses `X-Forwarded-For` itself rather than through `TRUST_X_FORWARDED_FOR`. **It finds an address only when the header carries exactly one**; with several it finds none and logs a warning. Three of these five uses need `TRUST_X_FORWARDED_FOR=true` (or Vercel) on top of that, as [the frontend](#the-frontend) describes.

| What reads the address                                                                              | Needs `TRUST_X_FORWARDED_FOR` | With no address                                            |
| --------------------------------------------------------------------------------------------------- | ----------------------------- | ---------------------------------------------------------- |
| The [authentication API's own limits](rate-limits.md#the-authentication-api)                        | No                            | One shared count for every caller                          |
| The [sign-in and password-reset attempt limits](rate-limits.md#signing-in-and-resetting-a-password) | Yes                           | Only the per-email-address limits apply                    |
| The [per-IP cap on account emails](rate-limits.md#account-emails)                                   | No                            | The cap doesn't apply; the per-address ones still do       |
| The sign-in history under **Admin → Users**                                                         | Yes                           | No address is recorded                                     |
| IP blocks                                                                                           | Yes                           | No address to match; blocks on an email domain still apply |

**Set `X-Forwarded-For` to the visitor's address rather than appending to it**, and set `TRUST_X_FORWARDED_FOR=true` on the frontend. Caddy replaces the header by default; in nginx the line is `proxy_set_header X-Forwarded-For $remote_addr;`. Without it, two things can go wrong:

- nginx's `$proxy_add_x_forwarded_for` appends the visitor's address to whatever arrived. A browser sends no `X-Forwarded-For`, so one nginx gives exactly one entry — but a second proxy in the chain, or a request that already carried the header, makes it two, and the limits above fall back to one count.
- With no `proxy_set_header X-Forwarded-For` line at all, nginx passes the header through as it arrived, so whatever the request carried is what Better Auth reads.

The backend deletes the sign-in history of an account after 90 days, or after the number of days in `AUTH_IP_EVENT_RETENTION_DAYS` (1 to 3650).

## What to set

| Your install                                              | On the frontend                                                                                  | On the backend                                                                                                                                          |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| AutoGPT Cloud                                             | Nothing                                                                                          | Nothing                                                                                                                                                 |
| Single-container image                                    | Nothing                                                                                          | Nothing                                                                                                                                                 |
| Docker Compose, browsers connect to the frontend directly | Leave `TRUST_X_FORWARDED_FOR` unset                                                              | Nothing                                                                                                                                                 |
| Behind a reverse proxy                                    | `TRUST_X_FORWARDED_FOR=true`, if the proxy sets `X-Forwarded-For` to the visitor's address alone | `FORWARDED_ALLOW_IPS=<the address it connects from>` and `TRUSTED_PROXY_COUNT=<entries your proxies add>`, if the proxy is also in front of the backend |

On any install that serves the public, put the frontend behind a reverse proxy that sets `X-Forwarded-For` to the visitor's address alone, replacing any value the request arrived with.
