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

## Link published advisories from a release

The helper reads only published advisories and checks that both the patched
version and an exact release link match. Withdrawn reports are excluded. It
includes accepted researcher credits and the CVE when assigned; a CVE is not
required for a link.

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
