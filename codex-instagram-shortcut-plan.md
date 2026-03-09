# Scope

This project defines a compliant iPhone Shortcut workflow to save Instagram videos to Photos **only** when the video is from your own account, or from a public post where you have explicit permission and the save/download use is allowed by Instagram terms and the source content rights. Private/protected content, bypassing access controls, and any scraping/reverse-engineering of private APIs are explicitly out of scope.

# Recommended Approach

Use a **Share Sheet + user-provided URL + optional local helper service** pattern:

1. User shares an Instagram post URL to a Shortcut named with `codex`.
2. Shortcut asks compliance questions (ownership/permission/rights).
3. Shortcut sends the URL plus attestation flags to a local helper service you control.
4. Helper service performs policy checks and only processes allowed URLs.
5. Helper returns a direct downloadable media file URL or the video bytes.
6. Shortcut saves to Photos using native iOS action.

If helper cannot legally/technically fetch media in a compliant way, Shortcut stops and informs user.

# Repo Name

`codex-instagram-save-shortcut`

# Shortcut Name

`Save to Photos codex`

# Architecture

## Option A: Shortcut-only

- Input: Share Sheet URL or clipboard URL.
- Validation: domain format + compliance prompts.
- Output: Save file to Photos.

Tradeoffs:
- Simplest UX.
- Limited processing capability.
- Hard to implement robust legal/compliance logging.

## Option B (recommended): Shortcut + local helper server

- iPhone Shortcut handles UX and permissions.
- Local helper (e.g. `http://192.168.x.x:8787`) handles policy enforcement and optional media retrieval pipeline.
- Local-first means no third-party cloud dependency by default.

Tradeoffs:
- Slight setup complexity.
- Better maintainability, logs, and strict policy gating.

## Option C: Shortcut + GitHub-hosted docs only

- Repo stores instructions and template Shortcut import notes.
- No running service.

Tradeoffs:
- Safest and easiest to share.
- No automation beyond manual workflow.

## Recommended components

- **iPhone Shortcut**:
  - Receives URL from Share Sheet.
  - Asks explicit questions:
    - "Is this from your own account?"
    - "If not, do you have explicit permission and rights-compliant use?"
  - Stops if either answer is non-compliant.
  - Calls helper endpoint.
  - Saves response to Photos.

- **Helper service (optional but recommended)**:
  - Endpoint: `POST /v1/resolve`
  - Input: URL + attestations.
  - Enforces allowlist + attestation requirements.
  - Returns either:
    - `{ status: "ok", download_url: "..." }` or
    - `{ status: "blocked", reason: "..." }`

- **Compliance log** (local JSONL):
  - Timestamp, URL hash, attestation flags, decision.
  - No private token storage.

# Implementation Phases

## Phase 0 — Policy and scope lock

- Define in-scope content matrix.
- Define blocked scenarios (private accounts, no permission, unclear rights).
- Add clear legal notice in README.

## Phase 1 — Repository bootstrap

Create repo structure:

- `README.md`
- `docs/policy.md`
- `docs/shortcut-setup.md`
- `docs/troubleshooting.md`
- `shortcut/Save-to-Photos-codex.steps.md`
- `helper/` (if using local service)
  - `app.py` or `server.ts`
  - `policy.py` or `policy.ts`
  - `requirements.txt` or `package.json`

## Phase 2 — Build Shortcut MVP

- Trigger from Share Sheet and clipboard fallback.
- URL normalization and domain checks.
- Compliance prompts.
- Stop with clear message if not compliant.
- If compliant, call helper or continue manually.

## Phase 3 — Build local helper

- Implement strict request schema.
- Validate attestations before any retrieval attempt.
- Enforce domain allowlist (`instagram.com`, `www.instagram.com`).
- Return blocked reasons for non-compliant inputs.

## Phase 4 — Save-to-Photos integration

- Shortcut receives media URL/data.
- Uses `Get Contents of URL` + `Save to Photo Album`.
- Handles errors with actionable user messages.

## Phase 5 — Test and harden

- Positive tests for allowed cases.
- Negative tests for blocked cases.
- Network timeout/retry behavior.
- Verify no private tokens or bypass logic exists.

# Shortcut Flow

1. **Receive input** from Share Sheet.
2. **Extract URL**.
3. **Validate URL** host is Instagram public URL format.
4. **Ask menu**:
   - Own content? (Yes/No)
   - If No: explicit permission + rights compliance? (Yes/No)
5. If not compliant => **Show Result** and **Stop Shortcut**.
6. If compliant => call helper `POST /v1/resolve` with JSON payload.
7. If helper returns `ok` => download media and save to Photos.
8. Show success/failure summary.

# Helper Service Design

## Why helper is useful

- Centralizes policy checks.
- Easier to update compliance logic.
- Optional local logs for audits.

## Local-first deployment options

- Laptop on same Wi-Fi as iPhone.
- Raspberry Pi home server.
- Local container with static LAN IP.

## API contract

### Request

```json
{
  "url": "https://www.instagram.com/reel/...",
  "attestation": {
    "is_owner": false,
    "has_explicit_permission": true,
    "rights_compliant": true
  }
}
```

### Response (allowed)

```json
{
  "status": "ok",
  "download_url": "https://.../video.mp4"
}
```

### Response (blocked)

```json
{
  "status": "blocked",
  "reason": "Private/protected or insufficient rights attestation"
}
```

# Sample Code / Pseudocode

## Shortcut pseudo-logic

```text
ON share_sheet_input(url):
  if not is_instagram_url(url):
    alert("Only Instagram URLs are supported")
    stop

  own = ask_yes_no("Is this video from your own Instagram account?")

  if own == NO:
    permitted = ask_yes_no("Do you have explicit permission and rights-compliant authorization to save this public video?")
    if permitted == NO:
      alert("Blocked: permission/rights requirement not met")
      stop

  response = http_post(helper + "/v1/resolve", {
    url,
    attestation: {
      is_owner: own,
      has_explicit_permission: permitted_or_true_if_owner,
      rights_compliant: permitted_or_true_if_owner
    }
  })

  if response.status != "ok":
    alert("Blocked: " + response.reason)
    stop

  media = http_get_binary(response.download_url)
  save_to_photos(media)
  alert("Saved to Photos")
```

## Python helper sketch (FastAPI)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

app = FastAPI()

class Attestation(BaseModel):
    is_owner: bool
    has_explicit_permission: bool
    rights_compliant: bool

class ResolveRequest(BaseModel):
    url: HttpUrl
    attestation: Attestation


def is_allowed(req: ResolveRequest) -> bool:
    if req.attestation.is_owner:
        return True
    return req.attestation.has_explicit_permission and req.attestation.rights_compliant


@app.post("/v1/resolve")
def resolve(req: ResolveRequest):
    host = req.url.host.lower()
    if host not in {"instagram.com", "www.instagram.com"}:
        return {"status": "blocked", "reason": "Unsupported host"}

    if not is_allowed(req):
        return {"status": "blocked", "reason": "Insufficient attestation"}

    raise HTTPException(status_code=501, detail="Implement a compliant retrieval path or manual import fallback")
```

# Testing Plan

## Functional tests

1. Own-account public reel URL + `is_owner=true` => should pass policy gate.
2. Public URL + explicit permission true => should pass policy gate.
3. Public URL + no permission => blocked.
4. Non-Instagram URL => blocked.
5. Private/protected URL or inaccessible resource => blocked/fail-safe.

## Failure cases

- Helper offline.
- Invalid JSON response.
- Timeout during media fetch.
- Photos permission denied.
- URL copied but malformed.

## Validation checklist

- No token/session scraping logic.
- No private account workflows.
- No anti-bot bypass instructions.
- Clear user prompts for rights attestation.

# Risks and Constraints

- Platform terms and copyright law vary by region; user must verify lawful use.
- Even with permission, some content may still be restricted by platform policy.
- Private/protected content is always out of scope.
- If no compliant retrieval route exists, use manual export methods supported by Instagram and the original creator.

# Final Checklist

- [ ] Repo created as `codex-instagram-save-shortcut`
- [ ] Shortcut created as `Save to Photos codex`
- [ ] Scope statement included in README/docs
- [ ] Policy gate prompts implemented in Shortcut
- [ ] Local helper endpoint running (optional)
- [ ] Photos permission granted on iPhone
- [ ] Negative tests for non-compliant scenarios pass
- [ ] Legal notice added: allowed vs disallowed content
