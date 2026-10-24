# Security advisory credits and release notes

The GitHub advisory is the source of truth for each vulnerability, its affected
versions, first fixed version, CVE and credits. A release note links to that
record rather than copying the full report. Publishing a release does not
publish its private advisories.

## Backfill an advisory

1. Verify the fix in the release tag, not only on `dev`. Record the first release
   containing the fix. Distinguish a partial mitigation from a complete fix.
2. Add a link to that release in the advisory's Fix or Release references section.
   Use numeric versions in package metadata; explain the corresponding Git tag
   in the description. Do not replace the first fixed version with the latest
   release used for validation.
3. Preserve researcher names, consent and duplicate-report contributions. Use
   GitHub's Credits field and the researcher's chosen handle/profile link. Do not
   put email addresses in public acknowledgments without explicit consent.
4. After publication, update `SECURITY.md` with accepted reporter, finder and
   analyst credits. Count distinct published advisories per person, not credit
   roles or duplicate submissions. Preserve existing historical acknowledgments.
   Pending or declined invitations and unpublished reports are not inputs to the
   public backfill.

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

The helper reads only published advisories and checks that both the patched
version and an exact release link match. Withdrawn reports are excluded. It
includes accepted researcher credits and the CVE when assigned; a CVE is not
required for a link.

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

The helper does not edit GitHub or publish advisories. It replaces only its
`security-advisories` marker block, preserving other release notes. With no
matching published advisories, or malformed markers, it fails without writing
the output. Rerun it after additional advisories are published, including those
published after their fixing release. Check release-edit workflow triggers
before applying notes to an existing release.

Run its fixture checks with:

```sh
python -m unittest discover -s .github/workflows/scripts -p test_security_release_notes.py
```
