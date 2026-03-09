# Known Issues & Limitations — Claude-InstaSaver

This document tracks known limitations, Instagram policy considerations, and technical
constraints affecting Claude-InstaSaver. Check this file before opening a bug report.

---

## Instagram Terms of Service

> **Important**: Instagram's [Terms of Use](https://help.instagram.com/581066165581870) (§ 1)
> state that users may not "collect users' content or information, or otherwise access Instagram,
> using automated means without our prior permission." Downloading videos with this shortcut
> may violate these terms.
>
> **Use responsibly**: Only download content you have the right to save — your own videos,
> or content where the creator has given explicit permission. Do not redistribute downloaded
> content without authorisation.

The shortcut is provided for **educational and personal archiving purposes only**. The authors
accept no liability for how it is used.

---

## Known Technical Limitations

### 1. Private Account Videos

**Status**: Partially working
**Severity**: Medium

Method A (Instagram internal API) returns data only for posts the currently-authenticated
browser session can access. Since the Shortcuts app has no Instagram session cookie, private
account posts will return an error response even if you follow that account.

**Workaround**: None available without authentication. A future version could accept a session
cookie as an input variable.

---

### 2. Instagram API Endpoint Changes

**Status**: Ongoing risk
**Severity**: High

The `?__a=1&__d=dis` endpoint is an **undocumented, internal** Instagram API. Instagram can
change, rate-limit, or remove it at any time without notice.

**Signs it has changed**:
- Method A always fails even for public posts
- The API returns HTML instead of JSON

**Workaround**: If Method A breaks, the shortcut falls through to Method B (fallback service).
If both fail, check GitHub Issues for an updated version of `generate_shortcut.py`.

---

### 3. Fallback Service Availability

**Status**: External dependency
**Severity**: Medium

Method B relies on `sssinstagram.com`, a third-party service. This service:
- May impose rate limits (typically after several requests per hour from the same IP)
- May change its API endpoint or response format
- May go offline permanently

**Workaround**: Wait and retry. If persistently broken, edit `generate_shortcut.py` and
replace the fallback URL with an alternative service, then regenerate the `.shortcut` file.

---

### 4. Stories Are Not Supported

**Status**: By design
**Severity**: Low

Instagram Stories have a different URL format (`/stories/username/media_id/`) and expire
after 24 hours. The regex in this shortcut does not match story URLs.

**Workaround**: Not currently implemented. Story downloading would require a separate
authentication-aware approach.

---

### 5. Carousel Posts (Multiple Videos)

**Status**: Partial support
**Severity**: Low

For carousel posts containing multiple videos, the shortcut currently extracts only the
**first video** from the post's item array (`items.0.video_url`).

**Workaround**: Run the shortcut multiple times with the same URL — Instagram's API may
return different items depending on the session context. A full carousel download loop is
planned for v1.1.

---

### 6. Rate Limiting by Instagram

**Status**: Active risk
**Severity**: Medium

Instagram rate-limits anonymous API requests. After downloading several videos in a short
period, subsequent requests may receive HTTP 429 (Too Many Requests) or 403 (Forbidden)
responses.

**Signs of rate limiting**:
- Method A consistently fails for a period of time
- Requests time out instead of returning an error

**Workaround**: Wait 10–15 minutes between bulk downloads. Using a VPN may help reset rate
limits, though this may itself violate Instagram's ToS.

---

### 7. Large Video Files

**Status**: iOS constraint
**Severity**: Low

The Shortcuts app downloads the entire video file into memory before saving to Photos.
For very long videos (>5 minutes, >200 MB), this may:
- Cause the shortcut to time out
- Use significant device RAM
- Fail on older devices with limited memory

**Workaround**: Not directly fixable within Shortcuts. For very large files, consider using
a desktop tool instead.

---

### 8. iOS Version Compatibility

**Status**: Tested on iOS 15–17
**Severity**: Low

The shortcut targets iOS 15+ (`WFWorkflowMinimumClientVersion: 900`). Actions used:
- `is.workflow.actions.text.match` (iOS 13+)
- `is.workflow.actions.savephotosalbum` (iOS 13+)
- `is.workflow.actions.notification` (iOS 13+)

No iOS 18 compatibility issues have been identified, but Shortcuts app internals change
between major iOS versions.

---

### 9. Shortcut File Trust Warning

**Status**: By design (Apple policy)
**Severity**: Informational

Apple requires shortcuts distributed outside the Shortcuts Gallery to be manually trusted
by the user (**Settings → Shortcuts → Allow Untrusted Shortcuts**). This is a deliberate
Apple security measure, not a bug in Claude-InstaSaver.

---

## Reporting New Issues

Please open a GitHub Issue with:
1. The iOS version and Shortcuts app version
2. The type of Instagram URL (reel, post, IGTV)
3. Whether the account is public or private
4. The exact error message displayed by the shortcut
5. Whether Method A or Method B was attempted (check the shortcut run log)
