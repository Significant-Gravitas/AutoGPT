# Swap proxy

The proxy every AutoPilot sandbox egresses through. Code the model writes never
holds a real credential: it holds a placeholder such as `hsurr:github`, and this
proxy replaces the placeholder with the user's real token on the way out, in
the `Authorization` header only, and only for requests going to a host that
credential is bound to.

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
4. **Swap.** A placeholder in the `Authorization` header becomes the owner's
   value: `Bearer` and `token` (curl, `gh`) and HTTP Basic, which is what git
   sends, also for a token in the remote URL (git and curl turn URL userinfo
   into this header; it never goes out as part of the URL). A placeholder
   anywhere else (another header, the path, the query, a body, a websocket
   message) goes out literally and is audited as `refused` /
   `outside-authorization`. A body is where a provider stores things: a value
   swapped into one could be written to a gist, an issue or a blob and read
   back later in whatever encoding the provider offers (base64, hex, a git
   packfile), which no scrub can match. A value in `Authorization` is used,
   not stored. The swap everywhere else is kept (`swap_anywhere` on a
   credential) for providers that need it, and arrives with the binding
   table (SECRT-2616); nothing sets it yet. The backend is asked for values
   per user and per host, and refuses hosts a credential is not bound to.
   (`swap_proxy/swap.py`, `swap_proxy/source.py`)
5. **Scrub.** A value echoed back in a text response, or in a websocket message
   from the server, is turned back into its placeholder before the box sees it,
   and so is the exact base64 of an HTTP Basic pair the proxy built, which a
   server echoing the header would send back. The values scrubbed are the
   user's current ones for that host plus any swapped into that very request. The host is the one the request and the
   connection name, proven or not: over plain http too, since removing a value
   never sends one. If the backend cannot say what the user's
   values are, a text response with a body is refused (`refused-response` /
   `resolver-unavailable`) and a server websocket message dropped
   (`refused-message`), rather than passed on unscrubbed: a value can come back
   in a response that did not ask for it, from wherever it is stored at the
   provider.
   The scrub also runs over the body's bytes, with each value as UTF-8 and as
   UTF-16, whatever charset the body declares: a body that does not decode as
   declared is scrubbed that way instead of being let through, and one the box
   reads in another charset than the proxy did is covered too.
6. **Only what can be scrubbed.** For a box that gets swaps: no swap for
   `TRACE` / `TRACK`, whose response is the request itself (`message/http`,
   not a type the scrub reads); a `101` that is not a websocket is refused
   (`unscrubbable-upgrade`); and anything but HTTP inside a connection the
   proxy opened is closed (`refused-connection` / `not-http`) rather than
   relayed as raw bytes, which nothing scrubs. A swap also needs an HTTP/1
   absolute-form target, when one is sent, to name the verified host. Plain
   protocols the proxy never opens (ssh in the clear, say) still pass.

Every swap, scrub and refusal is one JSON line on the `swap_proxy.audit` logger,
with names and reasons, never values. A swap is recorded only for bytes that had
not left yet.

### Large bodies

The proxy holds at most 5 MiB of one body in memory (`MAX_BODY_BYTES` in
`addon.py`, the only size in the service); mitmproxy streams anything larger.
A box chooses how much it sends and, by what it asks for, how much comes back,
so the size must never decide whether a credential is protected:

| | up to 5 MiB | over 5 MiB, or growing past it |
| --- | --- | --- |
| request head (the `Authorization` header) | swapped | swapped, before anything is sent |
| text request body | not swapped (audited); with `swap_anywhere`, swapped | streamed, not swapped; with `swap_anywhere`, sent as it is and audited `body-not-swapped` |
| text response body | scrubbed | **refused**: the flow is killed; audited `refused-response` |
| binary body, either way | streamed, untouched | streamed, untouched |

The 5 MiB counts bytes on the wire, and a compressed body stands for more, so a
held body with a `Content-Encoding` has a second limit on what it may decode to:
20 MiB (`MAX_DECODED_BYTES`, four times the wire limit). The limit is enforced
while decoding (`decode.py` asks each decoder for at most that much), never by
decoding the whole body and measuring it, so a few kilobytes of gzip standing
for gigabytes cost the proxy no more than 20 MiB and are then turned away:

| a held text body that… | request (with `swap_anywhere` only) | response |
| --- | --- | --- |
| decodes to more than 20 MiB | sent as it is; audited `body-not-swapped` / `decoded-too-large` | **refused**; audited `refused-response` / `decoded-too-large` |
| is in an encoding that cannot be decoded within a bound (anything but `gzip`, `deflate`, `br`, `zstd`, or more than one), or is corrupt | sent as it is; audited `body-not-swapped` / `undecodable-encoding` | **refused**; audited `refused-response` / `undecodable-encoding` |

A request body the proxy will not decode may or may not name a credential;
nobody can say without decoding it. It goes out untouched, which is the safe
direction, and the head of the request is still swapped. A response the proxy
cannot read is one the box could still decode, so it is not passed on. Both
only apply to a box that gets swaps and a user with a credential for the host:
for anyone else there is nothing to swap or scrub and the body is not read.

So a `git push` with a large pack authenticates (its body is binary and
streams), and a text response too large to scrub never reaches the box. A
request body is held back only for a credential with `swap_anywhere`; by
default nothing can be swapped into it, and it streams. A body with no declared
length (chunked, HTTP/2) that is held is held until it ends or passes the
limit, so a value split across chunks is still caught.

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
| `SWAP_PROXY_CONFDIR` | `~/.mitmproxy` | directory the signing CA is mounted into; see below |
| `SWAP_PROXY_GENERATE_CA` | `false` | local runs only: generate a CA if the directory has none |
| `SWAP_PROXY_EGRESS_ALLOW` | empty | comma-separated private hosts or CIDRs boxes may reach anyway |
| `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`, `REDIS_CLUSTER_HOST`, `REDIS_CLUSTER_PORT`, `REDIS_USE_ANNOUNCED_ADDRESS` | | same meaning as in the backend |

### The signing CA

The boxes' image trusts one CA certificate, and the proxy signs the certificate
of every bound host with that CA's key. So:

- **One CA for every replica and every restart.** A replica that signed with a
  CA of its own would break TLS to every bound host inside every box, and it
  would read there as a certificate bug, not as missing provisioning.
- **Provisioned, never generated.** Create the key pair once, keep it in your
  secret store, and mount it into `SWAP_PROXY_CONFDIR` as `mitmproxy-ca.pem`
  (private key and certificate in one PEM file). If the mount is read-only, put
  `mitmproxy-dhparam.pem` beside it: it is not a secret (fixed public
  parameters mitmproxy writes out with every CA), but mitmproxy tries to create
  it when it is missing. The service **refuses to start** without the CA rather
  than let mitmproxy make one up. `SWAP_PROXY_GENERATE_CA=true` lifts that for
  a local run; `mitmproxy-ca-cert.pem` in the directory is then the certificate
  to trust.
- **The key lives in the secret store and in the running proxy, nowhere else.**
  Whoever holds it can read the traffic of every bound host. It is not in the
  image, the repository or a box.
- **Rotation is a coordinated change**: boxes must trust the new certificate
  before the proxy starts signing with the new key.

### Turning it on

The backend only sends boxes here once `E2B_EGRESS_PROXY_ADDRESS` is set, and
E2B fails closed. Setting that address and provisioning the CA are **one release
step**: the proxy reachable at that address, the CA mounted into every replica,
and its certificate in the image the boxes run. Any one of them missing shows up
as every box losing either its egress or its TLS to bound hosts.

The same step sets the network policy around the proxy, and that policy is a
security boundary, not housekeeping. The proxy calls one backend service
(`SWAP_PROXY_BACKEND_URL`, the internal `DatabaseManager`) and must be able to
reach that and nothing else in the backend: the policy is what keeps a
compromised proxy away from every other backend service.

It cannot do more than that. `DatabaseManager` serves every method it exposes
on the same port, with no caller authentication or per-caller allowlist, and
the proxy uses only two of them (`get_swap_bindings`,
`resolve_swap_credential`). A proxy that can reach it can call all of them,
reading any user's credentials among them. Restricting it to the two swap
methods needs a dedicated service or a per-caller allowlist (SECRT-2742);
until then that is an accepted gap, and one more reason the proxy's own host
is to be treated as holding every user's credentials.

### What one message costs the event loop

The swap and the scrub run synchronously on mitmproxy's event loop, which every
connection of a replica shares. Measured on a developer laptop (Apple silicon,
Python 3.13), one body at the 5 MiB limit, median of several runs:

| work | time |
| --- | --- |
| swap a JSON request body | 23 ms |
| scrub a JSON or plain-text response | 2-3 ms |
| scan a gzip response with nothing to scrub (decode included) | 8 ms |
| scrub a gzip response that did echo a value (decode, scrub, re-encode) | 53 ms |

That is how long every other connection on the replica waits while one such
message is handled; a box can cause it at will, one message at a time. These
are single-machine numbers, not a capacity figure: how many boxes a replica
carries has not been measured, and server CPUs will differ. The table is for
5 MiB of decoded text. A compressed body may decode to up to 20 MiB, so the
worst single message costs about four times the gzip rows.

## Tests

```bash
poetry run pytest
```

`e2e_test.py` and `e2e_tls_test.py` run the real proxy on loopback with a
hand-written SOCKS5 client in front and a local server behind; they are where
the tenant boundary is tested (wrong, stale and borrowed credentials). The
rest are unit tests, `swap_test.py` being the port of spark-vm's conformance
suite. `addon_test.py` drives the hooks with mitmproxy's own test flows for what
a hand-written HTTP/1 client cannot reach (websockets, HTTP/2 framing).

CI is `.github/workflows/platform-swap-proxy-ci.yml`. Its `pull_request`
trigger only fires for a base of `master`/`dev`/`release-*`; for a stacked PR
run it by hand: `gh workflow run platform-swap-proxy-ci.yml --ref <branch>`.

## Not here yet

- Per-user bindings and more providers: the binding table is
  `SUPPORTED_PROVIDERS[...]["swap_hosts"]` in the backend, GitHub only.
- Time-limited grants and the approval step that spark-vm has.
- E2B tunnels TCP only: DNS and QUIC leave a box without passing through here.

## Known limits

- Images and binary bodies are neither swapped (above 5 MiB) nor scrubbed (at
  any size). What counts as text is the content type the sender declares.
- A value goes into the `Authorization` header and nowhere else. A provider
  that takes its key in another header, the query or a body cannot be used
  until the binding table opts that credential in (`swap_anywhere`,
  SECRT-2616).
- Only literal values are scrubbed, and the exact Basic pairs the proxy built.
  The same value in another encoding (base64, hex, inside an archive) is not
  recognised. With the swap confined to `Authorization`, no value is written
  into a request body through the proxy, which is how one would come to be
  stored at a provider and served back so encoded; one stored there by other
  means (by the user, say) is not caught.
- A text response over 5 MiB from a bound host is refused, not delivered, for a
  box that gets swaps.
- A compressed text response from a bound host that decodes to more than
  20 MiB, or uses an encoding other than `gzip`, `deflate`, `br` or `zstd` (or
  several at once), is refused for a box that gets swaps; a request body like
  that is not swapped (with `swap_anywhere`; otherwise it is never swapped).
- The decoded-size limit covers HTTP bodies. Websocket messages are
  decompressed by mitmproxy before the addon sees them, with no limit of ours.
- A text response with no declared length is held until it is complete, so an
  event stream from a bound host does not arrive incrementally.
- Websocket messages are swapped and scrubbed one message at a time; a value
  split across two messages is not recognised.
- While the backend cannot be asked, a box that gets swaps receives no text
  response with a body and no server websocket message from a bound host
  (each refused and audited), and nothing is swapped into its requests beyond
  the 15 s a fetched value stays cached. Binary responses still pass, as they
  are never scrubbed. Boxes that do not get swaps are unaffected. If the
  proxy has never had an answer from the backend since it started, it cannot
  tell bound hosts from others: a swapping box's TLS connections are then
  opened whatever their host, and the same refusals apply to all of them until
  the backend first answers.
- Only bound hosts are scrubbed. A value stored at a provider and served back
  from a host that is not bound (a raw-content domain, say) passes through
  unread. The name is the box's to
  choose: TLS to a provider's address under an SNI that is not bound, or plain
  http with such a `Host`, is the same case. For a box that gets swaps, TLS
  with no SNI at all and plain http to a bare address are refused
  (`refused-connection` / `refused-request`, reason `no-sni`); that narrows
  the case, it does not close it.
- A body in gzip or zstd may hold at most 64 members or frames; more is
  treated as undecodable.
- A connection stays authenticated while it stays open, also after its box's
  credential is rotated.
- NAT64 (`64:ff9b::/96`, and its local-use prefix) and 6to4 addresses are judged
  by the IPv4 address they carry. Teredo and operator-chosen NAT64 prefixes are
  not recognised.
