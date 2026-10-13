# AutoGPT Platform Load Tests

[k6](https://grafana.com/docs/k6/latest/) tests that drive traffic at a deployed platform: a `!deploy` preview, dev, or a local stack. The backend reports every request to Sentry, and the [perf-compare skill](../../../.claude/skills/perf-compare/SKILL.md) compares runs there.

Docker is the only requirement. `./k6.sh` runs k6 from the pinned `grafana/k6` image with this directory mounted, and takes the same arguments as `k6`.

## Quick start: a preview

1. Comment `!deploy` on the pull request. A few minutes later its backend answers at `https://pr-<N>-server.agpt.co/health`, and its frontend is `https://autogpt-pr-<N>.vercel.app`.
2. Run a test against it:

   ```bash
   cd autogpt_platform/backend/load-tests
   export K6_ENVIRONMENT=PREVIEW PR_NUMBER=<N>
   VUS=2 DURATION=1m RAMP_UP=10s RAMP_DOWN=10s ./k6.sh run tests/api/core-api-test.js
   ```

Nothing else is needed: on a preview, the tests sign up their own throwaway accounts (see [Accounts](#accounts)).

## Quick start: dev

The whole team shares dev, so keep the load small and tell the team first. The tests cannot create accounts on dev; they sign in with accounts you supply (see [Accounts on dev](#dev)):

```bash
cd autogpt_platform/backend/load-tests
export K6_ENVIRONMENT=DEV
export LOAD_TEST_ACCOUNTS="$(cat configs/accounts.json)"   # gitignored
VUS=2 DURATION=1m RAMP_UP=10s RAMP_DOWN=10s ./k6.sh run tests/api/core-api-test.js
```

## Tests

| Test                                           | Auth | What it does                                                                                   |
| ---------------------------------------------- | ---- | ---------------------------------------------------------------------------------------------- |
| `tests/basic/connectivity-test.js`             | yes  | Backend `/health`, the frontend's JWKS, and one authenticated request                          |
| `tests/basic/single-endpoint-test.js`          | yes  | One endpoint (`ENDPOINT=credits\|graphs\|blocks\|executions`)                                  |
| `tests/api/core-api-test.js`                   | yes  | `/api/credits`, `/api/graphs` and `/api/blocks` together                                       |
| `tests/api/graph-execution-test.js`            | yes  | Creates small graphs and executes them (402 counts as handled when the account has no credits) |
| `tests/marketplace/public-access-test.js`      | no   | Store browsing: agents, search, categories, creators                                           |
| `tests/marketplace/library-access-test.js`     | yes  | Library listing and search; adds a store agent to the library when the store has one           |
| `tests/comprehensive/platform-journey-test.js` | yes  | Credits, graphs, blocks and executions with think time                                         |

A preview starts with an empty database: store and library lists come back empty, so costs that grow with data do not show there.

## How authentication works

The backend accepts only the ES256 JWTs that the frontend's Better Auth issues (audience `authenticated`), and verifies them against `<frontend>/api/auth/jwks`. `lib/auth.js` gets them the way the browser does:

- `setup()` signs each account in once per run (`POST <frontend>/api/auth/sign-in/email`, or `sign-up/email` for throwaway accounts) and exchanges the session for a token (`GET <frontend>/api/auth/token`). Every VU then uses one account's token, so a run signs in once per account however many VUs it has.
- Better Auth lets one IP make 3 sign-in or sign-up requests, then refuses more until 10 seconds pass without one. `setup()` sends them in bursts of 3 with an 11-second pause, and waits out any `429`. Five accounts take about 15 seconds.
- Tokens last an hour. Each VU fetches a new one with its account's session 10 to 20 minutes before the old one expires, so runs can last longer than an hour.
- Before the load starts, `setup()` sends one authenticated request and stops the run unless it succeeds.

Passwords, cookies and tokens stay in memory and are never logged.

## Accounts

### Previews

With `LOAD_TEST_ACCOUNTS` unset, `setup()` signs up `LOAD_TEST_ACCOUNT_COUNT` (default 5) throwaway accounts named `loadtest-<run>-<n>@previews.agpt.co`, with random passwords that never leave the run. This works because previews allow sign-up for `@previews.agpt.co` and do not require email verification, so no email is sent and no shared secret is needed. The accounts stay in the preview's database until the preview is torn down.

To run as the seeded personas instead (`preview-existing@previews.agpt.co` and the others), set `LOAD_TEST_ACCOUNTS`. Their password is the `PREVIEW_ACCOUNTS_PASSWORD` secret in the team Bitwarden:

```bash
export LOAD_TEST_ACCOUNTS="$(jq -n --arg p "$PREVIEW_ACCOUNTS_PASSWORD" \
  '[{email: "preview-existing@previews.agpt.co", password: $p}]')"
```

### Dev

Dev's sign-up allowlist does not include `@previews.agpt.co`, and `@agpt.co` addresses can only sign up through Google, so the tests cannot sign up on dev. Running the authenticated tests there needs:

- A few dedicated email-and-password accounts. An admin can create them with Better Auth's admin API (`POST https://dev-builder.agpt.co/api/auth/admin/create-user`, signed in as an admin), or a domain for them can be added to dev's `AUTH_SIGNUP_ALLOWLIST` (`DEV_AUTH_SIGNUP_ALLOWLIST` in the infra repo's dev CD workflow) so they can sign up.
- Verified email addresses, if dev requires email verification.
- Their passwords in the team secret store, handed to the tests as `LOAD_TEST_ACCOUNTS`.

### `LOAD_TEST_ACCOUNTS`

A JSON array of accounts that already exist on the target:

```json
[{ "email": "loadtest-1@example.com", "password": "..." }]
```

The tests read it only from the environment. To keep it between runs, save it as `configs/accounts.json` (gitignored) and `export LOAD_TEST_ACCOUNTS="$(cat configs/accounts.json)"`. The tests sign in each account in the list once; VUs share them round-robin.

### Local stack

`K6_ENVIRONMENT=LOCAL` targets `http://localhost:8006` and `http://localhost:3000`; inside Docker, `./k6.sh` points it at `host.docker.internal`. The tests sign up throwaway accounts unless the local frontend restricts sign-up.

## Environment variables

`./k6.sh` passes these into the container by name when they are set in your shell, so values never appear on a command line.

| Variable                                                     | Used for                                                                                  |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| `K6_ENVIRONMENT`                                             | `PREVIEW`, `DEV`, `LOCAL` or `PROD` (default `DEV`)                                       |
| `PR_NUMBER`                                                  | The pull request whose preview `PREVIEW` targets                                          |
| `LOAD_TEST_ACCOUNTS`                                         | Accounts to sign in with; required on `DEV` and `PROD`                                    |
| `LOAD_TEST_ACCOUNT_COUNT`                                    | Throwaway accounts to sign up when `LOAD_TEST_ACCOUNTS` is unset (default 5, at most 100) |
| `VUS`, `DURATION`, `RAMP_UP`, `RAMP_DOWN`                    | Load shape                                                                                |
| `REQUESTS_PER_VU`                                            | Requests per VU per iteration                                                             |
| `ENDPOINT`, `CONCURRENT_REQUESTS`                            | `single-endpoint-test.js` only                                                            |
| `THRESHOLD_*`                                                | Threshold overrides in the marketplace tests                                              |
| `K6_ALLOW_PROD`                                              | Must be `true` for `PROD`; agree a production run with the team first                     |
| `K6_CLOUD_TOKEN`, `K6_CLOUD_STACK_ID`, `K6_CLOUD_PROJECT_ID` | Grafana Cloud k6 runs only                                                                |
| `K6_IMAGE`                                                   | Another k6 image for `./k6.sh` (default `grafana/k6:2.3.0`)                               |
| `K6_BIN`                                                     | A k6 binary for the orchestrator to use instead of `./k6.sh`                              |

## Results in Sentry and perf-compare

k6 prints its own summary at the end of a run. For comparisons, use the backend's traces in Sentry:

| Target  | Sentry project                                                     | Environment                      |
| ------- | ------------------------------------------------------------------ | -------------------------------- |
| Preview | `autogpt-server-previews` (backend), `builder-previews` (frontend) | `pr-<N>`                         |
| Dev     | `autogpt-server` (backend), `builder` (frontend)                   | `app:dev-behave:cloud` (backend) |

Each route is a transaction named by its path, such as `/api/credits` or `/api/graphs/{graph_id}/execute/{graph_version}`. `/health` is not traced, and the sign-in requests from `setup()` go to the frontend project. Sentry's `failure_rate()` counts the 402 an account without credits gets from graph execution, although k6 does not. Note the UTC start and end of each run (`date -u +%FT%TZ`), wait a few minutes for Sentry to ingest the last events, then compare windows with perf-compare. For example, a candidate preview against a control preview under the same load:

```bash
S=$(git rev-parse --show-toplevel)/.claude/skills/perf-compare/scripts/perf_compare.py
python3 "$S" --project autogpt-server-previews --transaction '/api/credits' \
  --baseline env=pr-<CONTROL> window=<START>/<END> \
  --candidate env=pr-<N> window=<START>/<END>
```

perf-compare wants at least 200 samples from 30 traces per side, and recommends 1000 requests per transaction. The skill covers choosing baselines, interleaving runs and reading the verdict.

## Orchestrator

`orchestrator/orchestrator.js` runs all twelve scenarios in turn, at 100 to 500 VUs each, for about an hour: a stress suite rather than a comparison run. It needs Node and, by default, Docker:

```bash
PR_NUMBER=<N> node orchestrator/orchestrator.js PREVIEW          # local, through ./k6.sh
LOAD_TEST_ACCOUNTS="$(cat configs/accounts.json)" node orchestrator/orchestrator.js DEV
```

It checks its settings before the first test, and writes a summary to `results/`.

## Grafana Cloud k6 (optional)

Runs can also execute on Grafana Cloud k6 (project `4254406`) with a Grafana Cloud k6 API token and the stack's numeric ID:

```bash
export K6_CLOUD_TOKEN=... K6_CLOUD_STACK_ID=... K6_CLOUD_PROJECT_ID=4254406
./k6.sh cloud run --env K6_ENVIRONMENT=PREVIEW --env PR_NUMBER=<N> --env VUS=50 tests/api/core-api-test.js
PR_NUMBER=<N> node orchestrator/orchestrator.js PREVIEW cloud
```

A remote cloud run sees only the variables passed with `--env`, and Grafana Cloud keeps those with the run. So the orchestrator refuses to send `LOAD_TEST_ACCOUNTS` there; cloud runs suit previews, where the tests sign up throwaway accounts. Results appear at <https://significantgravitas.grafana.net/a/k6-app/>.

## Troubleshooting

- **`DEV needs LOAD_TEST_ACCOUNTS`**: dev and prod have no throwaway sign-up; see [Accounts on dev](#dev).
- **`Signing up throwaway account ... failed: HTTP 403`**: the target does not allow sign-up for `@previews.agpt.co`. Use `LOAD_TEST_ACCOUNTS`.
- **`Getting a token ... failed: HTTP 401` right after sign-up**: the target requires email verification, so a sign-up gets no session. Use verified accounts in `LOAD_TEST_ACCOUNTS`.
- **`... rejected the token from ...`**: the backend verifies tokens against a different frontend's JWKS than the one the test signed in at.
- **`The first authenticated request ... failed`**: the backend is down, still starting, or not where `K6_ENVIRONMENT` and `PR_NUMBER` point.
- **`429` during setup**: another sign-in from the same IP used up the burst. `setup()` waits and retries; it fails only after five refusals in a row.
