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
   links without maintaining counts. Historical acknowledgments stay outside
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

1. The read-only job checks out trusted `master` code and produces only a
   snapshot digest and source commit. It also checks release reconciliation
   without writing.
2. The credits job uses that exact code commit, re-fetches anonymous public
   records, and requires the digest to agree. It treats `dev` as data, generates
   the two allowed files, and opens or updates one PR against `dev`. The existing
   bot PR branch is never executed. Only the PR action receives its write token.
3. The release job independently re-fetches and compares public records, then
   rechecks the projection after reading release metadata. It reads each target
   release anonymously immediately before updating its managed section. Drafts
   and immutable releases are not edited. It preserves all other notes from that
   fresh read. GitHub does not document conditional PATCH support for releases:
   an edit racing the final read and write can still be overwritten. Avoid manual
   release-note edits during this short job, and inspect concurrent changes before
   rerunning. Updates are serialized but are not a cross-release transaction.

The release writer uses only the repository's short-lived `GITHUB_TOKEN` with
`contents: write`. The credits writer additionally needs `pull-requests: write`.
No personal access token or advisory-write credential is used. The authenticated
transport can only PATCH the body of an existing release; there is no operation
that creates or publishes an advisory or release. GitHub does not start release
workflows for release edits made with this token. Do not replace it with a PAT:
older release tags may retain workflows that build images on `release.edited`.
The current Classic workflow also limits itself to published Classic tags.

Public release sections contain advisory links, CVEs and accepted credits, not
arbitrary report text or titles. When a mapping or published advisory is removed,
the next reconciliation removes its generated entry. If no entries remain, the
managed section says none are currently linked. Historical snapshots and release
history remain public history; withdrawal cannot erase previously published data.

### Enable after review

1. Merge the implementation through `dev` to `master`; the schedule and trusted
   code both run from `master`. Keep `PUBLIC_SECURITY_SYNC_ENABLED` unset initially.
2. Dispatch **Security - Public advisory sync** on `master` with `dry_run: true`.
   Check that it succeeds and the changed-release/immutable-release counts are
   expected. Local dry runs can use the same commands below.
3. Set the repository Actions variable `PUBLIC_SECURITY_SYNC_ENABLED` to `true`,
   then dispatch with `dry_run: false`. This enables the six-hourly PR and release
   writers. Leaving the variable unset disables both writers even when dispatch
   asks to apply changes.
4. Review the generated credits PR. GitHub may require a maintainer to approve
   CI workflows for a PR created with `GITHUB_TOKEN`; this automation does not
   approve or merge its own PRs. The repository must allow Actions to create PRs.

To stop writes, unset the variable. To recover from an API limit, changed digest,
concurrent edit or failed update, rerun the workflow; generation is idempotent.
Some release updates may already have succeeded before a later request fails.
Immutable release notes need a separate distribution channel, such as a link to
the repository's public advisories, because the workflow cannot edit them.

```sh
digest=$(python .github/workflows/scripts/security_public_sync.py digest)
python .github/workflows/scripts/security_public_publish.py --expected-digest "$digest"
```

Publication, verifying first fixed versions, duplicate consolidation and consent
remain maintainer decisions in the private advisory process. This workflow
cannot make them public. It also cannot prevent a person from accidentally
publishing sensitive content on GitHub; only approved, already public identifiers
are copied. Review changes to the workflow and generator as privileged code.

GitHub documents [unauthenticated access to public repository advisories](https://docs.github.com/en/rest/security-advisories/repository-advisories#list-repository-security-advisories),
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

## Link published advisories from a release

The manual helper reads published advisories anonymously and checks that both the patched
version and an exact release link match. Withdrawn reports are excluded. It
includes accepted researcher credits and the CVE when assigned; a CVE is not
required for a link.

Once automation is enabled, `security_public_sync.public_release_notes` owns the
shared `security-advisories` block. Each reconciliation replaces that block with
advisory links, CVEs and accepted credits, removing summaries generated by the
manual helper. Keep any prose that should persist outside the managed block.

An advisory with multiple package rows needs an explicit package-to-release
mapping in its description. Add one comment for each patched package, using
the exact ecosystem and name from that row, alongside the human-readable
release link:

```html
<!-- security-release: {"tag": "autogpt-platform-beta-v0.9.0", "package": {"ecosystem": "other", "name": "significantgravitas/autogpt"}} -->
```

The helper requires the mapped row's patched version to match the requested
release. A Classic row's patched version cannot establish a Platform fix, or
vice versa. Without a mapping, an otherwise matching multi-package advisory
stops generation so a maintainer can clarify it; it is not silently omitted.
Single-package advisories continue to use their release link and patched
version directly. The mapping records verified evidence, not an inferred
equivalence between source and Docker versions.

Export the release's current notes, then generate a separate candidate file:

```sh
gh release view autogpt-platform-beta-v0.9.0 --repo Significant-Gravitas/AutoGPT --json body --jq .body > release-notes.md
python .github/workflows/scripts/security_release_notes.py --tag autogpt-platform-beta-v0.9.0 --notes release-notes.md --output release-notes-updated.md
```

Review the diff, then an authorized maintainer can apply it:

```sh
gh release edit autogpt-platform-beta-v0.9.0 --repo Significant-Gravitas/AutoGPT --notes-file release-notes-updated.md
```

The manual helper does not edit GitHub or publish advisories. It replaces only its
`security-advisories` marker block, preserving other release notes. With no
matching published advisories, or malformed markers, it fails without writing
the output. Rerun it after additional advisories are published, including those
published after their fixing release. Check release-edit workflow triggers
before applying notes to an existing release.

Run its fixture checks with:

```sh
python -m unittest discover -s .github/workflows/scripts -p 'test_security_*.py'
```

The tests use synthetic records and private-data canaries. They check generated
files, stdout/stderr, pagination, rejected authentication/redirect paths, credit
consent, revalidation, preservation of unrelated notes, and repeated-run behavior.
