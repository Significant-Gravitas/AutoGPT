# Packaging stubs

These are starting-point manifests for the two third-party package managers
we plan to support post-PyPI-publish:

- `homebrew/autogpt-local-executor.rb` — Homebrew formula
- `scoop/autogpt-local-executor.json` — Scoop manifest

They are NOT wired up to a live tap / bucket yet — that is a repo-org-level
decision (which GitHub org, which tap name, who has push access). When the
project is ready:

1. Publish `autogpt-local-executor` to PyPI.
2. Fork these files into the chosen tap / bucket repo.
3. Fill in the `PLACEHOLDER-*` `url` / `sha256` / `hash` values from the
   live PyPI release.
4. Tag a release in this repo; CI in the tap/bucket repo should be wired
   to bump these manifests automatically on every new tag.

Until then, the primary install path stays:

```bash
pipx install autogpt-local-executor
```

per `docs/CROSS_PLATFORM.md` "Packaging & install" row.

## Release flow

`.github/workflows/release.yml` handles PyPI publish:

1. Bump `version` in `pyproject.toml` to match the tag you're about to push.
2. Tag + push: `git tag v0.0.1 && git push --tags`.
3. The workflow builds sdist + wheel, publishes to PyPI via trusted-publisher
   OIDC (no `PYPI_API_TOKEN` secret needed — register this repo + the
   `release` workflow + the `pypi` environment once at
   https://pypi.org/manage/account/publishing/), and creates a GitHub
   release with the dists attached.
4. After release lands, run `scripts/update_packaging.sh v0.0.1` locally
   to refresh the Homebrew formula + Scoop manifest in this repo with
   real url+sha256+version. The script fetches the published dists from
   PyPI, computes SHAs, and edits both files in place.
5. Copy the rendered files into the tap / bucket repos (paths printed
   by the script) and open PRs there.

Once a tap / bucket exists and we're publishing regularly, step 4 can be
moved into a separate workflow that auto-bumps the tap repo via a
cross-repo PAT. Not worth the complexity for v0.
