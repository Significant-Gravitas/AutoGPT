# Packaging stubs

These are starting-point manifests for the two third-party package managers
we plan to support post-PyPI-publish:

- `homebrew/autogpt-local-executor.rb` — Homebrew formula
- `scoop/autogpt-local-executor.json` — Scoop manifest

They are NOT wired up to a live tap / bucket yet — that is a repo-org-level
decision (which GitHub org, which tap name, who has push access). When the
project is ready:

1. Publish `autogpt-local-executor` to PyPI.
2. Fill in the `PLACEHOLDER-*` `url` / `sha256` / `hash` values from the
   live PyPI release (`scripts/update_packaging.sh` does this).
3. Generate the Homebrew formula's `resource` blocks (see step 5 of
   "Release flow" below). The formula does not work without them.
4. Copy these files into the chosen tap / bucket repo.
5. Tag a release in `autogpt-local-executor`; CI in the tap/bucket repo
   should be wired to bump these manifests automatically on every new tag.

The package is not on PyPI yet. Until its first PyPI release, install it
from the git repository, as [the README](../README.md) does:

```bash
pipx install git+https://github.com/Significant-Gravitas/autogpt-local-executor.git
```

After the first PyPI release, `pipx install autogpt-local-executor` works
too. See the "Packaging & install" table in `docs/CROSS_PLATFORM.md`.

## Release flow

`.github/workflows/release.yml` in the
`Significant-Gravitas/autogpt-local-executor` repo handles PyPI publish.
The copy of this package in the AutoGPT monorepo (`local_executor/`) has
no release workflow yet, so run these steps in `autogpt-local-executor`:

1. Bump `version` in `pyproject.toml` to match the tag you're about to push.
2. Tag + push: `git tag v0.0.1 && git push --tags`.
3. The workflow builds sdist + wheel, publishes to PyPI via trusted-publisher
   OIDC (no `PYPI_API_TOKEN` secret needed — register the
   `autogpt-local-executor` repo + the `release` workflow + the `pypi`
   environment once at https://pypi.org/manage/account/publishing/), and
   creates a GitHub release with the dists attached.
4. After release lands, run `scripts/update_packaging.sh v0.0.1` locally
   to refresh the Homebrew formula + Scoop manifest in this repo with
   real url+sha256+version. The script fetches the published dists from
   PyPI, computes SHAs, and edits both files in place. Running it again
   replaces the values from the previous run.
5. Generate the formula's `resource` blocks. This step is required:
   Homebrew installs the package with `pip --no-deps`, so without them
   pydantic, websockets, httpx, keyring and the rest are missing and
   `autogpt-shim` fails to start. It needs Homebrew:

   ```bash
   brew update-python-resources packaging/homebrew/autogpt-local-executor.rb
   ```

   Homebrew builds every resource from source, and pydantic-core needs
   Rust, so also add `depends_on "rust" => :build` to the formula. Redo
   this step on every release. The script warns while the formula has no
   resource blocks.
6. Copy the rendered files into the tap / bucket repos (paths printed
   by the script) and open PRs there.

Once a tap / bucket exists and we're publishing regularly, steps 4 and 5
can be moved into a separate workflow that auto-bumps the tap repo via a
cross-repo PAT. Not worth the complexity for v0.
