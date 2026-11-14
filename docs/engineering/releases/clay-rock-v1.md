# Clay & Rock V1

Repository baseline: `45275cbb0bb36aa9aadfd2688de94daab3a8456e` (public master inspected 23 September 2026).

Retargeted to `dev` at `bdb8806affd46e79d412163f94b903615cf677d9`. That branch expands the roster to 32 Experts; its additional roles and saved template cohorts need a follow-up copy/artwork review. Catalog defaults use the approved specialist artwork for Maria and Mina. The onboarding illustration also uses supplied transparent artwork for seven specialists.

## What changes

- Approved v1.1 neutral Otto, Maria and Mina artwork, shared across the product. All 72 image exports retain the source bytes; artwork uses contain sizing, density variants, PNG fallback and an accessible text fallback. No new body motion or expression assets.
- Template defaults and new hires receive the reviewed Maria/Mina appearance and corrected managed biographies. Existing hires keep their stored appearance and identity. Selection uses saved URLs, never a person's display name, job title, category or task state.
- Compact identities show names and functional roles without a repeated tagline; detailed specialist identities retain AI Expert disclosure. Includes Create an Expert wording and Appearance upload labels. Otto remains the personal Head of AI. Existing route names and tool identifiers remain compatible.
- Shared generation guidance for new Expert charters, onboarding recommendations and Expert chat: no seniority, human career, tenure or credential claims, and no blanket promises to replace people. This is prompt guidance, not proof that every model response will comply.
- Home shows the supplied status detail and next step. Expert work shows textual run guidance; detailed run badges use the same labels. Approval explanations can wrap instead of being visually clipped. Run enums and approval decisions are unchanged.
- Files navigation follows the Files page flag. Onboarding email source says Otto and links to `/copilot`; workflow, billing and unsubscribe destinations retain their purpose. Shared notification artwork uses versioned Otto PNGs.
- Expert appearance uploads accept PNG/JPEG/WebP up to 5 MB, retain file-signature validation and malware scanning, and receive fresh filenames. Creation and appearance updates no longer require AutoMod approval or Redis receipts. Updates retain authenticated ownership checks. Upload errors preserve saved appearances and creation drafts. Marketplace publication moderation is outside this change.
- Onboarding uses eight spaced nodes with transparent artwork and name-only labels. Transparent 512-pixel WebP assets total 1.11 MiB, down from 9.59 MiB of supplied PNGs. Otto supports explicit size and transparency choices; team cards and headers use the original circular artwork. Emails use a 320-pixel transparent PNG over their existing colored bands.

## Release dependencies and safe rollout

1. Ship the versioned public image files before, or together with, backend defaults and email templates. Older image paths remain available. Do not purge old assets referenced by existing users or emails.
2. Appearance uploads use the existing media storage and malware scanner. No AutoMod or approval-receipt configuration is required for them; existing marketplace moderation is unchanged.
3. No database migration or roster reseed is needed to show updated catalog defaults. The projection only replaces recorded old defaults on template rows; new hires persist the projected values. Existing hires are deliberately not rewritten by reads.
4. Do not run the roster seed as a cosmetic migration: it also synchronizes routines, preloads and other behavior. Saved avatars are excluded from reseed backfills, including unchanged old defaults. Its other presentation backfills read batches of 100 hires, compare each field with the previous template and guard each write against concurrent edits. Identity rescoping retains the existing recognized-default gate. Rescope retries use recorded pre-rescope presentation defaults when the template has already advanced, so interrupted runs resume without leaving legacy hires behind. Unknown legacy baselines still leave the hire unchanged rather than guessing which fields the owner edited. Database-backed migration tests still need the isolated integration stack.
5. Hire Experts, onboarding and Files remain separate feature cohorts. This change does not enable flags. Verify each intended release cohort before rollout.
6. Onboarding tour delivery is managed through MailerLite. Updating these repository templates does not update a live MailerLite campaign. Its source and published copy need a separate owner review.

No agent-run local browser tests or production data changes were made. The user manually reviewed onboarding, team, sidebar and chat pages. Only the local marketplace template catalog was populated for previewing; workflows were not seeded. Local DOM tests do not establish production availability. Production/development deployment parity and real email-client rendering remain release checks.

## V2, after review

- Approve artwork for the remaining roster Experts (30 after the merge from `dev`, up from 13 in the original audit); retain their existing assets for V1.
- Replace the legacy custom appearance picker and generator with the reviewed mineral library and its constraints. V1 does not claim that legacy Notion artwork meets Clay & Rock.
- Implement reviewed facial expressions while keeping the body still, with text, reason and next action remaining independently accessible.
- Add the broader appearance generation workflow, ownership/version metadata where needed, and cooldowns after product approval.
- Complete marketing/CMS and MailerLite artwork and copy work in their actual source systems.
- Consider a separately reviewed, reversible migration for untouched older hires; do not replace user-authored names, avatars, instructions or history.

## Validation record

The workspace `.context/` logs contain implementation checks. Current coverage includes saved appearance stability and fallbacks, template projection, conditional backfill, upload validation and scanning, avatar API compatibility, Files gating, creation flows and email destinations. Live credentials, schedules, permissions and billing were not exercised or changed.

The historical runs below predate the removal of appearance moderation; their moderation-specific tests were removed with that feature.

Completed local checks:

- Frontend formatting, lint and TypeScript: passed (existing lint warnings remain).
- Full frontend suite: 7,515 initially passed; the 14 failures were investigated and fixed. Final affected-suite run: all 321 tests across 22 files passed, including every initially failing file.
- Backend: 219 tests passed across presentation/backfill, moderation, avatar routes, media storage, notification rendering, onboarding recommendations and Expert context. Route tests used the existing API auth fixtures with a local test-user fixture; no database was started.
- Backend formatting/lint excluding the global type pass: passed. Type checking of every changed backend source file: zero errors.
- The repository-wide backend type pass reports four errors in untouched files: an optional signing-key type in `autogpt_libs/auth/jwt_utils_test.py:74`, and missing `playwright.sync_api` imports in three `test/fixtures/skills/webapp-testing/examples/` scripts. No Playwright/browser dependency was installed or run to resolve those examples.
- React Doctor: no reported issues in its changed-file scan (92/100).
- All 72 approved character image exports match the supplied pack byte for byte. `git diff --check` passed.

PR packaging keeps two original 1024-pixel PNG exports (Maria and Mina, each under 1 MB) through exact-path exceptions to the 500 KB added-file check. The manifest is excluded from the entropy scanner because its reviewed SHA-256 image checksums are public integrity metadata. Other files retain both checks, and the separate gitleaks scan remains enabled.

Review follow-up checks: 276 tool-schema checks, 120 Expert-tool/context tests, 104 avatar-route/moderation tests, 33 media/moderation tests, 6 presentation tests and 44 renderer tests passed. The full frontend CI suite passed on the initial PR; review fixes add flag-cohort, empty-title disclosure, create-upload query and error-recovery checks. React Doctor's latest scan reports one maintainability warning in the existing ThreadHeader component (84/100), with no security or correctness finding. Live deployment checks above remain outstanding.
