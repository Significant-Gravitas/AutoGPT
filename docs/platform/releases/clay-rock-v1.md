# Clay & Rock V1

Repository baseline: `45275cbb0bb36aa9aadfd2688de94daab3a8456e` (public master inspected 23 September 2026).

## What changes

- Approved v1.1 neutral Otto, Maria and Mina artwork, shared across the product. All 72 image exports retain the source bytes; artwork uses contain sizing, density variants, PNG fallback and an accessible text fallback. No new body motion or expression assets.
- Template defaults and new hires receive the reviewed Maria/Mina appearance and corrected managed biographies. Existing hires keep their stored appearance and identity. Selection uses saved URLs, never a person's display name, job title, category or task state.
- Functional role labels, explicit AI Expert disclosure, Create an Expert wording and Appearance upload labels. Otto remains the personal Head of AI. Existing route names and tool identifiers remain compatible.
- Shared generation guidance for new Expert charters, onboarding recommendations and Expert chat: no seniority, human career, tenure or credential claims, and no blanket promises to replace people. This is prompt guidance, not proof that every model response will comply.
- Home shows the supplied status detail and next step. Expert work shows textual run guidance; detailed run badges use the same labels. Approval explanations can wrap instead of being visually clipped. Run enums and approval decisions are unchanged.
- Files navigation follows the Files page flag. Onboarding email source says Otto and links to `/copilot`; workflow, billing and unsubscribe destinations retain their purpose. Shared notification artwork uses versioned Otto PNGs.
- New Expert image uploads are limited to PNG/JPEG/WebP, 5 MB, and require image moderation. A short-lived approval receipt is bound to the authenticated owner and uploaded URL. External URL changes cannot bypass review; unchanged existing URLs remain valid. Rejection or service failure leaves the saved appearance unchanged.

## Release dependencies and safe rollout

1. Ship the versioned public image files before, or together with, backend defaults and email templates. Older image paths remain available. Do not purge old assets referenced by existing users or emails.
2. Confirm the existing AutoMod endpoint accepts `type: image` with a base64 data URL and returns the documented approval response. Configure `automod_api_url`, `automod_api_key` and timeout; Redis must be available for approval receipts. Tests mock this service contract. Missing configuration or service errors block new uploads with a retry message; there is no fail-open bypass.
3. No database migration or roster reseed is needed to show updated catalog defaults. The projection only replaces recorded old defaults on template rows; new hires persist the projected values. Existing hires are deliberately not rewritten by reads.
4. Do not run the roster seed as a cosmetic migration: it also synchronizes routines, preloads and other behavior. Its presentation backfill now compares each field with the previous template and guards writes against concurrent edits. Identity rescoping retains the existing recognized-default gate. If a retry no longer has the original role/identity baseline, it defers that legacy hire entirely until that baseline is available, preserving a consistent older profile. Database-backed migration tests still need the isolated integration stack.
5. Hire Experts, onboarding and Files remain separate feature cohorts. This change does not enable flags. Verify each intended release cohort before rollout.
6. Onboarding tour delivery is managed through MailerLite. Updating these repository templates does not update a live MailerLite campaign. Its source and published copy need a separate owner review.

No production or development sessions were inspected, no browser tests were run, and no live data was changed. Local DOM tests do not establish production availability. Production/development deployment parity, responsive appearance, real moderation integration and email-client rendering remain release checks.

## V2, after review

- Approve artwork for the other 13 roster Experts; retain their existing assets for V1.
- Replace the legacy custom appearance picker and generator with the reviewed mineral library and its constraints. V1 does not claim that legacy Notion artwork meets Clay & Rock.
- Implement reviewed facial expressions while keeping the body still, with text, reason and next action remaining independently accessible.
- Add the broader appearance generation workflow, ownership/version metadata where needed, and cooldowns after product approval.
- Complete marketing/CMS and MailerLite artwork and copy work in their actual source systems.
- Consider a separately reviewed, reversible migration for untouched older hires; do not replace user-authored names, avatars, instructions or history.

## Validation record

The workspace `.context/v1-*.log` files contain the implementation checks. The final handoff records their results. Relevant coverage includes saved appearance stability and fallbacks, template projection, conditional backfill, owner-bound moderation receipts, rejection before storage, avatar API compatibility, Files gating, creation flows and email destinations. Live credentials, schedules, permissions and billing were not exercised or changed.

Completed local checks:

- Frontend formatting, lint and TypeScript: passed (existing lint warnings remain).
- Full frontend suite: 7,515 initially passed; the 14 failures were investigated and fixed. Final affected-suite run: all 321 tests across 22 files passed, including every initially failing file.
- Backend: 219 tests passed across presentation/backfill, moderation, avatar routes, media storage, notification rendering, onboarding recommendations and Expert context. Route tests used the existing API auth fixtures with a local test-user fixture; no database was started.
- Backend formatting/lint excluding the global type pass: passed. Type checking of every changed backend source file: zero errors.
- The repository-wide backend type pass reports four errors in untouched files: an optional signing-key type in `autogpt_libs/auth/jwt_utils_test.py:74`, and missing `playwright.sync_api` imports in three `test/fixtures/skills/webapp-testing/examples/` scripts. No Playwright/browser dependency was installed or run to resolve those examples.
- React Doctor: no reported issues in its changed-file scan (92/100).
- All 72 approved character image exports match the supplied pack byte for byte. `git diff --check` passed.

PR packaging keeps two original 1024-pixel PNG exports (Maria and Mina, each under 1 MB) through exact-path exceptions to the 500 KB added-file check. The manifest is excluded from the entropy scanner because its reviewed SHA-256 image checksums are public integrity metadata. Other files retain both checks, and the separate gitleaks scan remains enabled.
