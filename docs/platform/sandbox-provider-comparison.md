# Sandbox Provider Comparison: E2B vs Daytona

Two branches implement the same interactive desktop sandbox feature (see
`autogpt_platform/backend/backend/blocks/desktop/PARITY.md`) with different providers:

- `ntindle/sandbox-e2b-desktop` — E2B (existing integration + desktop/persistence/volumes)
- `ntindle/sandbox-daytona` — Daytona (full migration of the sandbox surface)

Because block IDs and schemas match, the standard test graph below runs unchanged on
either branch. Fill one column per provider from real runs and the `cost_meter` outputs.

## Standard test graph

1. `CreateDesktopSandboxBlock` (workspace_scope=user) → note cold-start time, stream URL
2. `DesktopCommandBlock`: `xdg-open https://example.com` (launches GUI browser)
3. `DesktopActionBlock`: screenshot
4. `DesktopActionBlock`: left_click at a known coordinate, screenshot_after=true
5. `DesktopActionBlock`: type "hello sandbox"
6. `DesktopFileBlock`: write `<workspace>/persist-test.txt`
7. `StopDesktopSandboxBlock`: suspend
8. `CreateDesktopSandboxBlock` again with the same scope (new sandbox) →
   `DesktopFileBlock`: read `<workspace>/persist-test.txt` (verifies volume persistence)
9. `CreateDesktopSandboxBlock` with `sandbox_id` from step 1 (verifies suspend/resume)
10. `StopDesktopSandboxBlock`: destroy

## Results

E2B column measured 2026-07-21 via `backend/scripts/desktop_smoke_test.py`.

| Metric | E2B | Daytona |
|---|---|---|
| Cold start → ready (s) | 3.3–3.9 (n=7) | |
| Stream ready, URL responds 200 (s) | 1.1–1.4 | |
| Action round-trip: click (s) | 0.2 typical (n=8 sandboxes). Caveat: during one ~40 min window, 5 consecutive sandboxes showed a uniform ~15.5 s per input action (queries unaffected); not reproducible since, cause undetermined (E2B-side transient suspected — uniform delay suggests an internal timeout) | |
| Action round-trip: screenshot (s) | 0.5–0.6 | |
| Suspend (s) | 0.4–0.8 | |
| Resume after suspend (s) | 0.6–0.8 (file state verified preserved) | |
| Volume file visible across sandboxes | not testable — volumes not enabled on account (private beta); blocks fall back to suspend-only persistence with warning | |
| $/h active (from cost_meter `rate_usd_per_hour_running`) | $0.166 (2 vCPU + 4 GiB) | |
| $/month suspended (published pricing) | $0 compute; storage retained (paused kept indefinitely) | |
| Volume storage pricing | private beta, not enabled | |
| Concurrent sandbox limit (plan) | 20 (Hobby) / 100+ (Pro) | |
| Max continuous session | 1 h (Hobby) / 24 h (Pro) | |

## Pricing

Full verified analysis with published-vs-assumed markers, source URLs, and five costed
scenarios: **[sandbox-provider-pricing.md](./sandbox-provider-pricing.md)** (researched
2026-07-22). Key conclusions:

- **Metered Linux compute rates are identical** on both providers: $0.000014/vCPU/s +
  $0.0000045/GiB-RAM/s ($0.1656/h for the reference 2 vCPU / 4 GiB desktop).
- The real cost difference is structural: E2B's desktop feature effectively requires
  **Pro at $150/mo fixed** (Hobby caps sessions at 1 h and concurrency at 20), while
  Daytona is pure pay-as-you-go (tier unlocks are spendable wallet top-ups, not fees).
  A single always-available desktop used 8 h/day lands at **~$179/mo on E2B vs ~$30/mo
  on Daytona**.
- Persistence at rest: E2B paused = $0 (kept indefinitely); Daytona stopped ≈
  $0.39–0.78/mo per desktop (disk only), archived = $0.
- Workspace volumes: Daytona free/GA (100 per org, subpath multi-tenancy); E2B private
  beta, unpriced.

| Non-pricing deltas | E2B | Daytona |
|---|---|---|
| Desktop support | `desktop` template (XFCE, x11vnc/noVNC) | `computer_use` API (Xvfb/xfce4/x11vnc/noVNC) + VNC access |
| Stream auth | VNC password embedded in URL | Signed preview URL (expiring) |

## Qualitative notes

- **Stream embed (E2B)**: the noVNC URL (password embedded as a query param) loads
  straight into the live desktop — HTTP 200, no interstitial or extra click. Contrast:
  Daytona preview URLs show an "I Understand, Continue" interstitial inside the iframe
  once per browser (see the Daytona branch's notes).
- **CoPilot integration (E2B)**: `start_desktop` uses a SEPARATE on-demand
  desktop-template sandbox (session-linked via Redis, auto-pauses when idle, resumes
  in ~1 s) rather than making the session sandbox desktop-capable — E2B bills actual
  vCPU+RAM per running second, so keeping every copilot session on the GUI template
  would tax sessions that never need a screen. Session files are tar-copied into the
  desktop once at creation (E2B has no cross-sandbox mounts until volumes GA).

- **E2B**: pause/resume preserves full machine state (RAM included) with ~1 s resume;
  volumes are private beta (blocks degrade gracefully to suspend-only persistence).
- **Daytona**: stop/start persists filesystem (not RAM); volumes are GA and free;
  computer-use API is first-class (input/screenshot/recording/accessibility endpoints)
  rather than xdotool-over-commands.
- Fill in observed stream quality, input latency feel, and any reliability events here.
