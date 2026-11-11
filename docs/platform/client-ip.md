# Client IP Addresses

## Overview

A few parts of the AutoGPT Platform count or record callers by their IP address:

- [Rate limits](rate-limits.md) that count per IP address: store search when signed out, signing in and account emails, and calls to the External API (v2) without a key.
- The sign-in history on an account's page under **Admin → Users**, which lists the IP addresses it signed up and signed in from.
- IP blocks, which an admin sets on that page to refuse sign-ups and sign-ins from an address.

This page explains where the backend and the frontend each get a caller's IP address, and what to set on a self-hosted install so it is the visitor's real one. On AutoGPT Cloud there is nothing to set.

## The backend

The backend uses the address of the connection it receives. When that connection comes from a proxy listed in `TRUSTED_PROXIES`, it reads the `X-Forwarded-For` header instead and takes the caller to be the first address from the right that the list doesn't include. Every request is read that way, the External API's per-IP limits included.

If a reverse proxy sits in front of the backend, set `TRUSTED_PROXIES` on the backend to the address it connects from — not necessarily the proxy's own address. A proxy on the host that reaches the backend through its published port (`8006` by default under Docker Compose) connects from the Docker network's gateway, not from `127.0.0.1`. Find it with `docker network inspect app-network --format '{{(index .IPAM.Config 0).Gateway}}'` (it varies per install; Docker assigns the subnet). A proxy running in its own container on the same network connects with its own container address instead.

The setting takes a comma-separated list of IP addresses or CIDR ranges, so `TRUSTED_PROXIES=172.18.0.0/16` covers a Docker subnet whose gateway address Docker may reassign. List every proxy in a chain: the backend walks back past each one it lists, so the proxies may append to the header or replace it. An entry that isn't an address or a range stops the backend from starting.

Once the backend trusts that address, let nothing else reach it there: the backend reads `X-Forwarded-For` from any connection that comes from a listed address. Under Docker Compose, publish the backend's port on the host's loopback only (`ports: ["127.0.0.1:8006:8006"]`), so only a proxy on the host can reach it, or drop the mapping and run the proxy in a container on the same network.

`TRUSTED_PROXIES` replaces two backend settings, which still work while it is unset and are ignored once it is set: `FORWARDED_ALLOW_IPS`, the server's (uvicorn's) own list of proxies, which defaults to `127.0.0.1,::1`, and `TRUSTED_PROXY_COUNT`, below. Both are deprecated, and the backend logs a warning naming `TRUSTED_PROXIES` when it reads either.

### Limits on the External API

The External API's per-IP limits, which only [API v2](rate-limits.md#api-v2) has, read the caller's address through `TRUSTED_PROXIES` like every other request.

While `TRUSTED_PROXIES` is unset, they fall back to the deprecated `TRUSTED_PROXY_COUNT` and take the `X-Forwarded-For` entry that many places from the right, whichever address connected, because everything further left is whatever the caller sent. Its default, `3`, is AutoGPT Cloud's own chain: Cloudflare appends the caller's address, and Google's load balancer appends two more. Without `TRUSTED_PROXIES`, set it to the number of entries your own proxies add, or `0` to count by the address the backend already uses, as [the backend](#the-backend) describes. When the header has fewer entries than the setting, the limit falls back to that same address; when the setting is higher than the entries your proxies add, it counts by an entry the caller can choose. This fallback takes the header from whatever connects, so with a setting above `0`, let callers reach the backend only through your proxies.

## The frontend

The frontend's server treats `X-Forwarded-For` as the visitor's address only when it runs on Vercel or when `TRUSTED_PROXIES` is set on the frontend. Without either, the frontend has no visitor address to pass on.

Set `TRUSTED_PROXIES` to the IP addresses or CIDR ranges of the proxies in front of the frontend, comma-separated, as on the backend. The visitor is then the first address from the right that the list doesn't cover, so the proxies may append to the header or replace it. An entry that isn't an address or a range stops the frontend's server from starting.

The frontend no longer reads `TRUST_X_FORWARDED_FOR`. When it is still set to `true` and the frontend has no `TRUSTED_PROXIES` and isn't on Vercel, the frontend's server logs a warning at start that it is ignored.

### The single-container image

The [single-container image](single-container.md) runs its own nginx in front of its frontend and backend, and points their `TRUSTED_PROXIES` at it. Behind a reverse proxy of your own, set `TRUSTED_PROXIES` on the container to that proxy's addresses: the image's nginx then reads the visitor's address through them, from a listed proxy only, and passes on only that address. Without it, every visitor appears as your proxy. An entry that isn't an address or a range stops the container from starting.

## From the frontend to the backend

Browsers reach the backend through the frontend's server, so the backend sees the frontend's address, not the visitor's. For store search, which the backend limits per IP address when the visitor is signed out, the frontend sends the visitor's address along in a signed token in the `X-Client-IP-Token` header, and the backend counts by that address.

- The token is signed with the frontend's sign-in key, which the backend already checks through `JWT_JWKS_URL`, and is valid for 60 seconds.
- When the frontend has no visitor address (see [the frontend](#the-frontend)), it sends no token. The backend then counts by the connection's address, so all signed-out store searches through one frontend server share one count.

## Signing in

Everything on the sign-in path reads the visitor's address through Better Auth, the frontend's sign-in library (1.7.16), which parses `X-Forwarded-For` itself. **Without `TRUSTED_PROXIES` it finds an address only when the header carries exactly one**; with several it finds none and logs a warning. With `TRUSTED_PROXIES` set, it takes the first address from the right that the list doesn't cover. Three of these five uses also need the frontend to trust the header (Vercel or `TRUSTED_PROXIES`), as [the frontend](#the-frontend) describes.

| What reads the address                                                                              | Needs the header trusted | With no address                                            |
| --------------------------------------------------------------------------------------------------- | ------------------------ | ---------------------------------------------------------- |
| The [authentication API's own limits](rate-limits.md#the-authentication-api)                        | No                       | One shared count for every caller                          |
| The [sign-in and password-reset attempt limits](rate-limits.md#signing-in-and-resetting-a-password) | Yes                      | Only the per-email-address limits apply                    |
| The [per-IP cap on account emails](rate-limits.md#account-emails)                                   | No                       | The cap doesn't apply; the per-address ones still do       |
| The sign-in history under **Admin → Users**                                                         | Yes                      | No address is recorded                                     |
| IP blocks                                                                                           | Yes                      | No address to match; blocks on an email domain still apply |

**List your proxies in `TRUSTED_PROXIES` on the frontend.** Without it, three of the five uses above get no address at all, and the other two find one only when the header carries exactly one entry, which goes wrong two ways:

- nginx's `$proxy_add_x_forwarded_for` appends the visitor's address to whatever arrived. A browser sends no `X-Forwarded-For`, so one nginx gives exactly one entry — but a second proxy in the chain, or a request that already carried the header, makes it two, and then no use finds an address (see the table's last column).
- With no `proxy_set_header X-Forwarded-For` line at all, nginx passes the header through as it arrived, so whatever the request carried is what Better Auth reads.

The backend deletes the sign-in history of an account after 90 days, or after the number of days in `AUTH_IP_EVENT_RETENTION_DAYS` (1 to 3650).

## What to set

| Your install                                              | On the frontend                                                              | On the backend                                                                                 |
| --------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| AutoGPT Cloud                                             | Nothing                                                                      | Nothing                                                                                        |
| Single-container image                                    | Nothing, or `TRUSTED_PROXIES` on the container behind your own reverse proxy | The container's `TRUSTED_PROXIES` covers both                                                  |
| Docker Compose, browsers connect to the frontend directly | Leave `TRUSTED_PROXIES` unset                                                | Leave `TRUSTED_PROXIES` unset and set `TRUSTED_PROXY_COUNT=0`                                  |
| Behind a reverse proxy                                    | `TRUSTED_PROXIES=<your proxies' addresses>`                                  | `TRUSTED_PROXIES=<the address it connects from>`, if the proxy is also in front of the backend |

On any install that serves the public, put the frontend behind a reverse proxy listed in its `TRUSTED_PROXIES`, and let browsers reach the frontend only through that proxy, because the frontend believes the header whoever sends it.
