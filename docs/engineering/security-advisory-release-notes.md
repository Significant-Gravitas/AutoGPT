# Security advisory credits and release notes

The GitHub advisory is the source of truth for each vulnerability, its affected
versions, first fixed version, CVE and credits. A release note links to that
record rather than copying the full report. Publishing a release does not
publish its private advisories.

## Backfill an advisory

1. Verify the fix in the release tag, not only on `dev`. Record the first release
   containing the fix. Distinguish a partial mitigation from a complete fix.
2. Add a link to that release in the advisory's description text, for example
   under a Fix or Release references heading. The helper searches the API's
   `description` field; a link only in `references[].url` is insufficient.
   Use numeric versions in package metadata; explain the corresponding Git tag
   in the description. Do not replace the first fixed version with the latest
   release used for validation.
3. Preserve researcher names, consent and duplicate-report contributions. Use
   GitHub's Credits field and the researcher's chosen handle/profile link. Do not
   put email addresses in public acknowledgments without explicit consent.
4. After publication, the public sync opens a PR updating `SECURITY.md` with
   accepted reporter, finder and analyst credits. It lists handles and profile
   links without maintaining counts, with a researcher's chosen name beside the
   handle when `RESEARCHER_NAMES` in `security_public_sync.py` has one (add one
   only when the researcher asks). Credits held by our own team, listed in
   `TEAM_LOGINS` in `security_release_notes.py`, are left out of both
   `SECURITY.md` and release notes. Historical acknowledgments stay outside
   the generated block. Pending or declined invitations and unpublished reports
   are never inputs to the public backfill.

## Public-only automation

`.github/workflows/security-advisory-sync.yml` runs every six hours and supports
manual dispatch on `master`. A schedule catches advisories published after their
fixing release, credit acceptance, corrections and withdrawals. It reconciles
all existing public releases, not just the latest one. The baseline is
`.github/security-advisories.public.json`, built entirely from anonymous public
API responses. Never initialize it from a private report inventory or export.

The snapshot contains only four fields per advisory: GHSA ID, optional CVE ID,
accepted researcher handles, and verified release tags. Every value must match
a restricted identifier format. Bodies, titles, discussions, attachments,
private-fork metadata, emails and pending credits are not stored or emitted.
The advisory body is examined in memory only to validate package-to-release
attribution. A changed snapshot produces added/updated/removed counts and a PR
diff; neither logs nor artifacts contain the raw response.

The reader uses standard-library HTTPS with no authentication, credentials,
proxy inheritance, redirects or authenticated retry. Pagination stays on this
repository's public endpoint, with page, response-size and timeout limits.
API failure, schema failure, incomplete pagination or ambiguous package
attribution stops the run. There is no fallback to stale snapshots.
Known failures log only a fixed, enumerated reason, such as a changed public
snapshot or an ambiguous release mapping. Exception text, chained errors and
response bodies are never printed; unexpected failures keep a generic message.

The jobs have separate permissions and no shared report artifacts:

1. The read-only job checks out trusted `master` code and passes only a
   snapshot digest and source commit to the writers. Its job summary previews
   the exact generated security sections and their public target releases, and
   names any published advisory that no published release lists. That happens when
   its description doesn't link the release that fixed it, its patched version
   doesn't match that release, or the release isn't published yet. The credits PR
   names them too. The preview contains no existing release text or raw advisory
   content.
2. The credits job uses that exact code commit, re-fetches anonymous public
   records, and requires the digest to agree. It treats `dev` as data, generates
   the two allowed files, and opens or updates one PR against `dev`. The existing
   bot PR branch is never executed. Both public Git checkouts run without
   credentials. After the files are generated, the job mints a batch-bot App token
   scoped to contents and pull requests, and only the PR action receives it.
3. The release job independently re-fetches and compares public records, then
   rechecks the projection after reading release metadata. It reads each target
   release anonymously immediately before updating its managed section. Drafts
   are not edited. Published immutable releases are eligible because GitHub
   permits updating their notes while protecting their tags and assets.
   The writer sends only the release body and `make_latest`, which is `true` for
   the current Latest release and `false` for every other release, because GitHub
   documents `make_latest` as defaulting to `true`. It reads `releases/latest`
   before the first edit and again after the last, and fails if the Latest
   release changed. It preserves all other notes from that fresh read. GitHub does not document conditional PATCH support for releases:
   an edit racing the final read and write can still be overwritten. Avoid manual
   release-note edits during this short job, and inspect concurrent changes before
   rerunning. Updates are serialized but are not a cross-release transaction.

Both writer jobs fetch their pinned public source without checkout actions or
credentials. GitHub permissions remain job-scoped, so their workflow code and
actions must be trusted. Both writers depend on successful validation and credit
preparation in the read-only job. Each has its own enablement switch; opening a
credits PR does not enable or approve release updates. A PR creation failure does
not block an independently enabled release writer.

The release writer uses only the repository's short-lived `GITHUB_TOKEN` with
`contents: write`. Its authenticated transport can only PATCH the body and Latest
flag of an existing release; there is no operation that creates or publishes an
advisory or release. GitHub does not start release workflows for release edits
made with this token. Do not replace it with a PAT or App token: older release
tags may retain workflows that build images on `release.edited`. The current
Classic workflow also limits itself to published Classic tags.

The credits writer is the opposite case. A PR opened with `GITHUB_TOKEN` gets its
CI runs parked until a maintainer approves them, so the credits PR is opened with
the batch-bot App's token instead (`BATCH_BOT_APP_ID`, `BATCH_BOT_PRIVATE_KEY`,
as the GitBook sync uses), and its CI runs like any other PR's. The job's own
`GITHUB_TOKEN` has no permissions. No personal access token or advisory-write
credential is used.

Public release sections contain advisory links, CVEs and accepted credits, not
arbitrary report text or titles. When a mapping or published advisory is removed,
the next reconciliation removes its generated entry. If no entries remain, the
managed section says none are currently linked. Historical snapshots and release
history remain public history; withdrawal cannot erase previously published data.

### Enable after review

1. Merge the implementation through `dev` to `master`; the schedule and trusted
   code both run from `master`. Keep `PUBLIC_SECURITY_CREDITS_ENABLED` and
   `PUBLIC_SECURITY_RELEASE_NOTES_ENABLED` unset initially.
2. Dispatch **Security - Public advisory sync** on `master` with `dry_run: true`.
   Review each target release and proposed security section in the job summary.
   The sections replace only the managed block; unrelated release text is omitted
   from the preview and preserved when writing. Local dry runs use the command below.
3. Set only `PUBLIC_SECURITY_CREDITS_ENABLED=true`, then dispatch with
   `dry_run: false`. This enables the credits PR writer without editing releases.
4. Review the generated credits PR. The batch-bot App opens it, so its CI runs on
   its own; it still needs an approving review, and this automation never
   approves or merges its own PRs. Its body names any published advisory that no
   published release lists.
5. After reviewing the public release preview, set
   `PUBLIC_SECURITY_RELEASE_NOTES_ENABLED=true` and dispatch with `dry_run: false`.
   Verify the first applied release sections, preservation of unrelated notes,
   and that the Latest release is still the newest release (the job checks it too).
   Subsequent scheduled runs reconcile releases automatically.

Both switches default to disabled. `dry_run: true` disables both writers regardless
of their switches. Each switch can be disabled independently.

To stop all writes, unset both switches. To recover from an API limit, changed digest,
concurrent edit or failed update, rerun the workflow; generation is idempotent.
Some release updates may already have succeeded before a later request fails.

```sh
digest=$(python .github/workflows/scripts/security_public_sync.py digest)
python .github/workflows/scripts/security_public_publish.py --expected-digest "$digest" --preview
```

Publication, verifying first fixed versions, duplicate consolidation and consent
remain maintainer decisions in the private advisory process. This workflow
cannot make them public. It also cannot prevent a person from accidentally
publishing sensitive content on GitHub; only approved, already public identifiers
are copied. Review changes to the workflow and generator as privileged code.

GitHub documents [unauthenticated access to public repository advisories](https://docs.github.com/en/rest/security-advisories/repository-advisories#list-repository-security-advisories),
[editable release notes on immutable releases](https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases),
[token permissions and event behavior](https://docs.github.com/en/actions/concepts/security/github_token),
and [scheduled workflow behavior](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).
See also the REST API's [conditional-request limitations](https://docs.github.com/en/rest/using-the-rest-api/best-practices-for-using-the-rest-api#use-conditional-requests).

## Consolidate duplicate reports

Compare the affected code path, missing control and attacker prerequisites.
Similar titles or a shared fixing commit do not establish a duplicate. Check
the submitted evidence when a title disagrees with the report body. Keep
separate weaknesses separate, even when an attack chains them together.

Choose one canonical advisory, summarize the overlapping evidence there, and
retain each reporter's credit. A report covering multiple weaknesses can
contribute evidence and credit to more than one canonical advisory. GitHub may
require recipients to accept newly added credits again; never represent a
pending invitation as accepted. Keep internal tool attribution in the source
record instead of assuming a product name identifies a researcher account.

Update the canonical record before closing a duplicate. Add a link from the
closed record to its canonical advisory and preserve the original evidence and
discussion. Closing a duplicate does not mean the vulnerability is fixed. Do
not put private report IDs, attachments or discussion into a public advisory.

## Identify Docker distributions

GitHub's [repository advisory API](https://docs.github.com/en/rest/security-advisories/repository-advisories#update-a-repository-security-advisory)
has no Docker ecosystem option. Use `other` and the image's repository name:

| Distribution | Package name | Registry |
| --- | --- | --- |
| Platform single-container image | `significantgravitas/autogpt` | [Docker Hub](https://hub.docker.com/r/significantgravitas/autogpt) |
| Classic image | `significantgravitas/auto-gpt` | [Docker Hub](https://hub.docker.com/r/significantgravitas/auto-gpt) |

Verify that the affected component ships in the image and that its runtime
configuration exposes the reported path. Source-based Compose host mappings
do not describe the single-container image's loopback bindings; generated
secrets can also change exposure. Record those prerequisites explicitly rather
than copying every source advisory into an image package row.

Use the image's own version history and an existing version tag. Record its
digest when pinning an exact artifact, because registry tags can be reassigned.
Explain the `v` prefix when package metadata uses numeric versions. `latest`
and development tags are mutable, so they are not fixed-version identifiers.
Keep Classic and Platform versions separate. A partial mitigation or an
unresolved issue has no fully patched image version; retain that distinction
in both the package metadata and the description.

## How an advisory is matched to a release

A published advisory is listed under a release when its description links that
release's exact tag URL and one of its package rows gives that release's version
as the patched version. Both are needed, so a release linked only for context,
such as where a bug was introduced or validated, isn't credited with the fix.
Withdrawn advisories are excluded, and a CVE is not required for a link.

When an advisory's package rows agree on the fixed version, such as the Platform
source and its Docker image both patched in 0.9.0, the release link and that
version are enough. When the rows report different patched versions, for
example a Classic row and a Platform row fixed in different releases, the
description needs an explicit package-to-release mapping. Add one comment for
each patched package, using the exact ecosystem and name from that row,
alongside the human-readable release link:

```html
<!-- security-release: {"tag": "autogpt-platform-beta-v0.9.0", "package": {"ecosystem": "other", "name": "significantgravitas/autogpt"}} -->
```

The helper then requires the mapped row's patched version to match the release.
Without a mapping, an advisory whose rows disagree stops generation so a
maintainer can clarify it; it is not silently omitted.

`security_public_sync.public_release_notes` owns the `security-advisories` block
in each release. Each reconciliation replaces that block with advisory links,
CVEs and accepted credits; keep any prose that should persist outside it. The
preview command above shows exactly what a run would write.

Run the fixture checks with:

```sh
python -m unittest discover -s .github/workflows/scripts -p 'test_security_*.py'
```

The tests use synthetic records and private-data canaries. They check generated
files, stdout/stderr, pagination, rejected authentication/redirect paths, credit
consent, revalidation, preservation of unrelated notes, and repeated-run behavior.
