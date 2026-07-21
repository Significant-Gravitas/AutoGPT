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
| Cold start → ready (s) | 3.9 | |
| Stream ready, URL responds 200 (s) | 1.4 | |
| Action round-trip: first click (s) | 16.8 (first xdotool call while XFCE settles; subsequent actions sub-second) | |
| Action round-trip: screenshot (s) | 0.5 | |
| Suspend (s) | 0.5 | |
| Resume after suspend (s) | 0.6 (file state verified preserved) | |
| Volume file visible across sandboxes | not testable — volumes not enabled on account (private beta); blocks fall back to suspend-only persistence with warning | |
| $/h active (from cost_meter `rate_usd_per_hour_running`) | $0.166 (2 vCPU + 4 GiB) | |
| $/month suspended (published pricing) | $0 compute; storage retained (paused kept indefinitely) | |
| Volume storage pricing | private beta, not enabled | |
| Concurrent sandbox limit (plan) | 20 (Hobby) / 100+ (Pro) | |
| Max continuous session | 1 h (Hobby) / 24 h (Pro) | |

## Published pricing snapshot (July 2026)

| | E2B | Daytona |
|---|---|---|
| Compute | $0.000014/vCPU/s + $0.0000045/GiB-RAM/s | per-vCPU/RAM/disk-hour, pay-as-you-go (see billing docs; Windows $0.0858/vCPU/h, Linux rates lower) |
| Suspended/stopped | Paused sandboxes free (storage retained; kept indefinitely) | Stopped sandboxes billed for disk only; archived to object storage is cheaper |
| Volumes | Private beta; storage-priced | Included free, up to 100/org |
| Plans | Hobby free ($100 one-time credits, 1h sessions, 20 concurrent) / Pro $150/mo (24h sessions, 100 concurrent) | Pay-as-you-go, $200 sign-up credits; startup program up to $50k credits |
| Desktop support | `desktop` template (XFCE, x11vnc/noVNC) | `computer_use` API (Xvfb/xfce4/x11vnc/noVNC) + VNC access |
| Stream auth | VNC password embedded in URL | Signed preview URL (expiring) |

## Qualitative notes

- **E2B**: pause/resume preserves full machine state (RAM included) with ~1 s resume;
  volumes are private beta (blocks degrade gracefully to suspend-only persistence).
- **Daytona**: stop/start persists filesystem (not RAM); volumes are GA and free;
  computer-use API is first-class (input/screenshot/recording/accessibility endpoints)
  rather than xdotool-over-commands.
- Fill in observed stream quality, input latency feel, and any reliability events here.
