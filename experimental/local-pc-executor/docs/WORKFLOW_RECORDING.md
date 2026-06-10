# Workflow Recording → Skill

> **Status**: Design draft v0.2 — revised after panel review. Not yet
> spec-locked.
>
> Record a workflow the user performs **anywhere on their machine**, hand
> it to the agent, and have the agent generalize it into a reusable
> **skill** that drives the user's computer to repeat it.
>
> Motivating example (one instance, not the ceiling): a user fills a
> browser form with one row of data by hand while recording; the agent
> turns that into a skill that fills the same form from every row of a
> CSV. The same machinery records a desktop app, a terminal sequence, or
> a flow that spans a browser *and* a native app.

---

## 0. What changed in v0.2 (and why)

v0.1 was browser-DOM-first, which forced a browser-only MVP and shipped
raw form values to the cloud for generalization. A review panel flagged
both as disqualifying. v0.2 re-founds the design:

1. **The floor is a computer-use trajectory, not a DOM trace.** Replay is
   done by our vision-capable computer-use agent re-planning against the
   live screen — not by deterministic selector playback. So the recording
   doesn't need perfect semantic selectors; it needs *enough for the agent
   to understand and redo the task*. That makes the floor universal
   (works in any app on any OS where computer-use works), with DOM/a11y as
   **enrichment**, not requirements.
2. **On-device interpretation is mandatory.** A general recording is
   screenshots of the user's whole screen. Those never leave the machine
   raw. The default path extracts structure on-device and sends only that
   to the cloud (see §3).
3. **Consent is shim-enforced, secret-detection is hygiene not a control,
   replay has explicit wait/assert/dedupe guards, and parameter inference
   must be confirmed (multi-row or asked), not guessed.** (Panel must-fixes,
   §7–§9.)

Capability target is unchanged and full: general desktop + browser
demonstration capture (the old "Tier 2") **and** the live co-pilot loop
with on-device interpretation (the old "Tier 3"). Those are the
baseline, not a phased cut. They are now framed as **two interaction
modes over one trajectory floor**, not two releases.

---

## 1. The floor: a computer-use trajectory

Every recording is an ordered list of **trajectory steps**. Each step is,
at minimum, *what the user did and what the screen looked like when they
did it* — the same shape computer-use models are trained on. Richer
channels (browser DOM, desktop accessibility) attach as optional
enrichment on the same step; they never replace the floor.

```jsonc
{
  "recording_id": "rec_<uuid>",
  "version": "1.0",
  "created_at": 1712345678.0,
  "machine_id": "...",
  "interpretation_route": "extract_then_cloud",  // see §3
  "steps": [ /* TrajectoryStep[] */ ],
  "redaction_applied": true
}
```

### `TrajectoryStep`

```jsonc
{
  "seq": 12,
  "ts": 1712345678.42,
  "actor": "human",                 // "human" | "agent"

  // --- the universal floor (always present) ---
  "action": "fill",                 // semantic verb (§1.1)
  "screenshot_ref": "stub_<uuid>",  // pre-action frame, stub id — NEVER inline bytes
  "cursor": [840, 314],             // where the action landed, display-global px
  "active_app": "Google Chrome",
  "active_window": "New customer — Acme",

  // --- enrichment: present only when the channel resolved it ---
  "enrichment": {
    "kind": "dom",                  // dom | ax | none
    "selectors": [                  // browser DOM only; most-stable-first (§7)
      {"strategy": "id",    "value": "#first_name"},
      {"strategy": "name",  "value": "first_name"},
      {"strategy": "label", "value": "First Name"},
      {"strategy": "role+text", "value": "textbox[name='First Name']"}
    ],
    "ax_path": null,                // desktop a11y only; element path within the AX tree
    "role": "textbox",
    "label": "First Name",
    "url": "https://app.example.com/new"
  },

  "value": {
    "raw": "John",                  // demo value — handled per interpretation route (§3)
    "type": "text",                 // text|email|number|date|secret|file|enum
    "is_parameter": null            // inferred + CONFIRMED during generalization (§8)
  },

  "narration": null,                // co-pilot mode: "filling the shipping address"
  "outcome": "ok",                  // ok | error | unknown
  "redacted": false
}
```

The key change from v0.1: `screenshot_ref` + `action` + `cursor` +
`active_app/window` are the **primary key** and are always present. The
`enrichment` block is additive — DOM selectors when the browser channel
saw them, an AX path when the desktop channel resolved them, `kind: none`
when neither did (pure visual step). Replay degrades gracefully: use the
DOM fast-path if present, the AX path if present, else visual grounding
on the screenshot.

### 1.1 Semantic verbs

`navigate`, `fill`, `select`, `click`, `submit`, `upload`, `launch_app`,
`focus_window`, `copy`, `paste`, `run` (shell), `file_op`, plus two
**replay-control verbs** the recorder synthesizes or the user adds (§9):
`wait` (for a condition — a spinner clears, an element appears) and
`assert` (a read-back check — "the confirmation banner showed").

**Not captured**: raw mouse movement paths, per-keystroke content (only
the final committed value), scroll deltas. The recording is a sequence of
*intentful semantic actions over screenshots*, not a raw input tape.

---

## 2. Two interaction modes (the capability baseline)

One trajectory floor, two ways the user can drive it. Both ship.

**Demonstration mode** (the old Tier 2). The user does the whole task by
hand; the shim captures the trajectory; the agent generalizes
afterward. Lowest friction, works when the task is unambiguous. Pairs
with a **multi-row** demonstration (§8) so parameter-vs-constant is
decidable.

**Co-pilot mode** (the old Tier 3, preferred default). The agent watches
the trajectory live, narrates each step, asks clarifying questions in
real time, and confirms with a dry-run before saving. Attacks the
dominant failure (confident misgeneralization) directly. This is the
default for anything non-trivial.

The user picks the mode per recording; the wire + schema are identical.

---

## 3. On-device interpretation (the privacy floor)

A general recording contains screenshots of the user's whole screen.
**Raw frames never leave the machine.** Interpretation — turning the
trajectory into an abstract skill — happens under one of three routes,
chosen per recording. The route is recorded in
`recording.interpretation_route`.

| Route | What leaves the machine | Capability | Privacy | Needs |
|---|---|---|---|---|
| `extract_then_cloud` **(default)** | text/structure only — OCR + a11y dump + DOM + window/app metadata per step; **no pixels, no raw secret values** | cloud-grade reasoning over extracted structure | high — pixels stay local | a local extractor (OCR is cheap; a11y/DOM already captured) |
| `local_vlm` (upgrade) | nothing — a local vision model authors the skill on-device; only the finished abstract skill is sent | capped by the user's local VLM | maximal — zero pixels, zero raw values leave | a capable local vision model (llava-class+) |
| `screenshots_to_cloud` (fallback) | screenshots, behind an explicit per-recording consent | maximal | lowest — frames leave the machine | nothing; gated on a hard consent the shim enforces |

**Default is `extract_then_cloud`** — it generalizes well *and* keeps
pixels on the machine. This is the privacy-mode pattern (local model /
pipeline extracts structure, strong cloud model reasons over it) applied
to screen recordings. The browser-DOM case is the happy path here: DOM is
already text, so extraction is free and lossless.

`local_vlm` is the zero-cloud upgrade for users with the hardware.
`screenshots_to_cloud` exists so users with no local extractor at all
aren't locked out — but it requires the shim-enforced consent gate (§9),
not a platform flag.

---

## 4. Capture adapters (universal floor + enrichment)

All adapters emit `TrajectoryStep`s into the same recording.

- **Screenshot+action floor (universal).** The shim's existing
  computer-use backends, run in reverse: instead of injecting input, the
  shim observes the user's action and snapshots the pre-action frame +
  cursor + active app/window. Works anywhere computer-use works
  (macOS/Windows/Linux-X11). This is the floor every other adapter
  enriches.
- **Browser DOM enrichment (lights up first).** A companion extension
  (extends the existing claude-in-chrome channel) attaches selectors +
  label + url to steps that happen in a browser tab, in the user's real
  logged-in session. Highest fidelity, and the one case where
  `extract_then_cloud` is lossless (DOM is text). Localhost-WebSocket
  transport to the shim (no per-OS native-host install for v1).
- **Desktop a11y enrichment.** macOS `AXObserver` / Windows UI Automation
  / Linux AT-SPI attach an AX path + role + label where the tree exists.
  Where it doesn't (Electron, canvas/WebGL, Wayland, games), the step
  keeps `enrichment.kind: none` and replay falls back to visual grounding
  — **not a failure, just lower fidelity.** This is why the floor matters:
  desktop coverage gaps degrade, they don't block.

**Rollout order is by tractability, not scope:** the floor + browser DOM
enrichment first (cheapest, most reliable, lossless privacy), desktop
a11y enrichment next. Both are in the design from day one; they light up
in sequence. The product is general from v1.

---

## 5. End-to-end: the motivating example (+ a desktop one)

**Browser form-fill:**
1. User picks co-pilot mode, clicks **Record** → shim-rendered indicator shows.
2. User fills the form once. Floor captures `fill`/`click` steps with
   screenshots; browser enrichment attaches `#first_name` etc.
3. Agent narrates live, asks "where does the data come from?" → "this
   CSV", and "stop on validation error?" → "skip the row."
4. User stops; the recording is interpreted via `extract_then_cloud`
   (DOM is already text — lossless). Agent confirms parameters
   (`first_name/last_name/email`) against the CSV columns.
5. Multi-row dry-run on rows 1–2 with the user watching → confirmed.
6. "Fill all 200 rows" → agent drives the browser via the DOM fast-path,
   with read-back asserts + per-row dedupe (§9).

**Desktop app (no DOM):**
1. User records renaming + tagging files in a native asset manager.
2. Floor captures screenshots + clicks; a11y enrichment resolves some
   buttons, leaves canvas regions as `kind: none`.
3. `extract_then_cloud`: on-device OCR + a11y dump → text per step →
   cloud generalizes "for each file, open it, set tag = {tag}, rename to
   {pattern}."
4. Replay: agent re-plans visually against the live UI, using a11y
   fast-paths where present, vision elsewhere.

Same machinery, fidelity scales with what the app exposes.

---

## 6. Wire ops

```jsonc
// platform → shim  (request/response — counts against in-flight)
{"type": "START_RECORDING", "payload": {
   "mode": "copilot",                      // "demonstration" | "copilot"
   "interpretation_route": "extract_then_cloud",
   "channels": ["floor", "browser", "desktop_ax"],
   "consent_token": "<shim-issued, see §9>"  // REQUIRED — platform can't self-assert
}}
{"type": "RECORDING_STARTED", "payload": {"recording_id": "rec_<uuid>"}}

{"type": "STOP_RECORDING",  "payload": {"recording_id": "rec_<uuid>"}}
{"type": "RECORDING_SUMMARY","payload": {"recording_id": "...", "step_count": 14,
   "enrichment_coverage": {"dom": 11, "ax": 0, "none": 3}, "duration_seconds": 47.2}}

{"type": "RECORDING_FETCH", "payload": {"recording_id": "..."}}
{"type": "RECORDING_DATA",  "payload": { /* full WorkflowRecording, post-redaction */ }}
```

**`RECORDING_STEP` is an unsolicited, non-acked, out-of-band stream**
(co-pilot mode only) — modeled like `STATUS`, **exempt from
`max_concurrent` / in-flight accounting and idempotency/retry** (the
panel's load-bearing protocol fix). Demonstration mode does **not**
stream; it buffers locally and the platform pulls via `RECORDING_FETCH`
after `STOP` + user approval — which is also what keeps demonstration-mode
data on the machine until the user consents to send it. (v0.1 contradicted
itself by streaming *and* claiming local-until-approved; resolved: stream
only in co-pilot mode, where live narration needs it and the user opted
into a live loop.)

Co-pilot interpretation reuses `LOCAL_LLM_COMPLETION` for the on-device
model; `INTERPRET_TURN` / `CLARIFY_PROMPT` / `DRY_RUN_REPLAY` as in v0.1
§3.5.

New error codes: `RECORDING_NOT_FOUND`, `RECORDING_CHANNEL_UNAVAILABLE`,
`RECORDING_ALREADY_ACTIVE`, `CONSENT_REQUIRED` (START without a valid
shim-issued consent token), `INTERPRETATION_UNAVAILABLE` (route needs a
local model the machine doesn't have — see §10).

---

## 7. Replay fidelity + selector strategy

Replay uses the richest available enrichment per step, falling through:

1. **DOM fast-path** — `selectors[]` tried most-stable-first
   (id > name > label > role+text > xpath); if exactly one node matches,
   use it; if zero or >1, fall through (no guessing).
2. **AX fast-path** — resolve the `ax_path` against the live tree; if it
   moved, fall through.
3. **Visual grounding** — the computer-use agent locates the target on the
   live screenshot the way it does for any computer-use task.

Because the floor is always a screenshot+action and replay is a vision
agent, **a broken selector degrades to visual replay rather than
failing.** That's the whole reason the trajectory floor buys robustness
the DOM-only design couldn't.

---

## 8. Parameter inference — confirmed, never guessed

A single demonstration can't distinguish parameter from constant. v0.2
**forbids shipping a skill with unconfirmed parameters.** Confirmation
comes from, in preference order:

1. **Co-pilot ask** — agent asks "does First Name change each run?"
2. **≥2 demonstration rows** — anything that changed is a parameter.
   Required for demonstration mode (no live loop to ask in).
3. **Data-source join** — match fields to named CSV/sheet columns by
   label similarity, then confirm.

Field semantics ("a field labeled Email is probably a parameter") is a
*prior* that seeds the question, **not** a basis for auto-save. A skill
whose parameters were never confirmed does not save — it blocks on a
clarifying question or a second row.

---

## 9. Consent, safety, security (panel must-fixes)

- **Shim-enforced consent (not platform-asserted).** `START_RECORDING`
  carries a `consent_token` the shim issues *only* after an OS-native,
  shim-rendered confirmation the platform cannot script. A platform that
  sends START without a valid token gets `CONSENT_REQUIRED`. The visible
  recording indicator is shim-rendered (tray/menu-bar), not the
  platform-controlled copilot UI.
- **Scoped capture.** Browser enrichment is **origin/active-form
  allow-listed** — it attaches to steps on the demonstrated origin, not
  every tab. The floor captures the active window only. No global
  keystroke hook.
- **Secret-detection is best-effort hygiene, explicitly NOT a privacy
  control.** Password fields / secret-shaped values are dropped, but the
  doc no longer claims "secrets never captured" as a guarantee — OTPs,
  account numbers, SSNs-in-generic-fields will slip a deny-list. The real
  control is `interpretation_route` (§3): pixels + raw values stay local.
- **Replay safety.** Every replayed step that mutates state gets a
  read-back `assert` (did the field hold the value? did the banner
  appear?); destructive sequences (`submit`) require a per-row **dedupe
  key** + **resume cursor** so a crash at row 150 doesn't re-file rows
  1–149; multi-row dry-run before any unattended run; a hard
  `destructive: true` flag on the generated skill.
- **At-rest.** The buffered recording is encrypted on disk under a
  **distinct** key in the OS keychain (not the audit-chain key — a
  different trust boundary), secure-erased after skill generation unless
  the user pins it.
- **Audit.** The audit log records *that* a recording happened + the
  channels + step count + interpretation route — never content. (Open:
  whether compliance needs a content manifest for data-subject requests,
  §10.)

---

## 10. Open questions (genuinely the author's / PM's call)

- **Default when no local extractor/VLM exists.** `extract_then_cloud`
  needs at least OCR + the already-captured a11y/DOM. If a machine has
  *nothing* (no OCR, no a11y, canvas-only app), does recording (a) block
  with `INTERPRETATION_UNAVAILABLE`, or (b) fall back to
  `screenshots_to_cloud` behind hard consent? Affects how many users v1
  reaches.
- **Parameter-confidence threshold.** What bar forces a clarifying
  question vs. accepts a confirmed-by-2-rows inference? Unenforceable
  until set.
- **Skill portability.** A recording on the user's Mac driving their
  local apps is machine-bound. Can a "file invoices in QuickBooks Online"
  (browser, portable) skill graduate into the shared library while a
  "rename files in Finder" stays shim-local? Two skill classes, or one?
- **Compliance manifest.** Does a content-free audit entry suffice, or do
  we need a reviewable (encrypted, local) manifest of what was captured to
  service deletion/access requests?
- **Desktop-app coverage floor.** For `kind: none` (canvas/Electron)
  steps, visual replay works but is slower + less certain. Is there a
  coverage threshold below which we tell the user "this app records
  poorly" up front?

---

## 11. References

- `COMPUTER_USE.md` — the input-injection backends this inverts for the
  capture floor; the a11y access this enriches with.
- `LOCAL_LLM.md` — the on-device interpretation channel (`extract_then_cloud`
  / `local_vlm`) routes through.
- `PRIVACY_MODE.md` — the extract-on-device-reason-in-cloud pattern §3
  generalizes to screen recordings.
- `AUDIT_LOG.md` — records *that* a recording happened, never its content.
- `PROTOCOL.md` — envelope, versioning, the in-flight/backpressure model
  `RECORDING_STEP` is exempted from, error-code base set.
