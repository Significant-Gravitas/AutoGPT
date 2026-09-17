# Swap proxy

The proxy every AutoPilot sandbox egresses through. Code the model writes never
holds a real credential: it holds a placeholder such as `hsurr:github`, and this
proxy replaces the placeholder with the user's real token on the way out, only
for requests going to a host that credential is bound to.

Ported from the credential-swapping proxy in spark-vm, adapted to serve many
users at once.

## How a connection goes

1. **Who is this?** The backend pins each E2B box's egress to this proxy with a
   per-box SOCKS5 username and secret (`backend/util/e2b_network.py`) and
   records the pair's owner in Redis. The proxy looks the username up and checks
   the secret against the stored digest. No match, no connection. The record
   also says whether this owner gets credentials swapped in at all: CoPilot
   boxes do; a block's box, which may run a graph someone else wrote, does not.
   (`swap_proxy/owners.py`)
2. **Where to?** The destination is resolved and refused if it lands in private
   address space; an allowed address is pinned so the check and the connect
   agree. (`swap_proxy/egress.py`)
3. **Open it or not?** Only hosts some credential is bound to are intercepted.
   Everything else is passed through as opaque bytes, never decrypted.
4. **Swap.** Placeholders in headers (including inside HTTP Basic), query, path
   and text bodies become the owner's values. The backend is asked for them per
   user and per host, and refuses hosts a credential is not bound to.
   (`swap_proxy/swap.py`, `swap_proxy/source.py`)
5. **Scrub.** A value echoed back in a text response is turned back into its
   placeholder before the box sees it.

Every swap and every refusal is one JSON line on the `swap_proxy.audit` logger,
with names and reasons, never values.

A value is only ever swapped into an https request whose upstream certificate
mitmproxy verified for the very host the request names. If the backend cannot
be asked, nothing is swapped: the placeholder goes out literally and the request
fails at the provider, loudly and with nothing leaked.

## Why a separate package

mitmproxy pins its dependencies exactly and needs Python 3.12+, and the E2B SDK
the backend depends on needs a newer `h2` than mitmproxy allows, so the two
cannot share an environment. The split is also the safer one: this is the
service that faces the internet and terminates TLS, and it has no database
access and no encryption key. It gets values from the backend one user and one
host at a time.

## Running it

```bash
poetry install
poetry run swap-proxy
```

| Variable | Default | |
| --- | --- | --- |
| `SWAP_PROXY_LISTEN_HOST` / `SWAP_PROXY_LISTEN_PORT` | `0.0.0.0` / `1080` | SOCKS5 listener |
| `SWAP_PROXY_BACKEND_URL` | `http://localhost:8005` | the backend's internal `DatabaseManager` service |
| `SWAP_PROXY_CONFDIR` | `~/.mitmproxy` | where the signing CA lives; created on first start |
| `SWAP_PROXY_EGRESS_ALLOW` | empty | comma-separated private hosts or CIDRs boxes may reach anyway |
| `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`, `REDIS_CLUSTER_HOST`, `REDIS_CLUSTER_PORT`, `REDIS_USE_ANNOUNCED_ADDRESS` | | same meaning as in the backend |

The boxes must trust the certificate in `SWAP_PROXY_CONFDIR`
(`mitmproxy-ca-cert.pem`). Its key must exist nowhere but here: whoever holds
it can read the traffic of every bound host.

The backend only sends boxes here once `E2B_EGRESS_PROXY_ADDRESS` is set. E2B
fails closed, so set it only when this proxy is reachable at that address.

## Tests

```bash
poetry run pytest
```

`e2e_test.py` and `e2e_tls_test.py` run the real proxy on loopback with a
hand-written SOCKS5 client in front and a local server behind; they are where
the tenant boundary is tested (wrong, stale and borrowed credentials). The
rest are unit tests, `swap_test.py` being the port of spark-vm's conformance
suite.

## Not here yet

- Per-user bindings and more providers: the binding table is
  `SUPPORTED_PROVIDERS[...]["swap_hosts"]` in the backend, GitHub only.
- Time-limited grants and the approval step that spark-vm has.
- Images and binary response bodies are not scrubbed.
- E2B tunnels TCP only: DNS and QUIC leave a box without passing through here.
