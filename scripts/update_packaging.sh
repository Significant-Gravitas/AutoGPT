#!/usr/bin/env bash
# Refresh packaging/homebrew/*.rb and packaging/scoop/*.json with the real
# url+sha256+version after a tagged release is published to PyPI.
#
# Usage:
#   scripts/update_packaging.sh v0.0.1
#
# Workflow:
#   1. Tag + push v0.0.1 → release.yml (in the
#      Significant-Gravitas/autogpt-local-executor repo) builds, publishes
#      to PyPI, attaches dists to the GitHub release.
#   2. Run this script locally with the tag.
#   3. Run `brew update-python-resources` on the formula to add its
#      dependency resource blocks. This script does not do that, and the
#      formula does not install a working shim without them.
#   4. Copy the rendered files into the homebrew tap / scoop bucket repo
#      and open a PR there.
#
# The repo's own packaging/ stubs are also updated in place so they stay
# usable for ad-hoc testing. Re-running the script replaces the values
# from the previous run.

set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: $0 <vX.Y.Z>" >&2
  exit 1
fi

TAG="$1"
VERSION="${TAG#v}"
PKG="autogpt-local-executor"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOMEBREW_FORMULA="$REPO_ROOT/packaging/homebrew/${PKG}.rb"
SCOOP_MANIFEST="$REPO_ROOT/packaging/scoop/${PKG}.json"

# Fetch sdist + wheel from PyPI and compute sha256s.
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

# Resolve the PyPI download URLs via the JSON API (avoids hardcoding the
# Files hosted layout, which has changed shape historically).
PYPI_JSON=$(curl -fsSL "https://pypi.org/pypi/${PKG}/${VERSION}/json")

SDIST_URL=$(echo "$PYPI_JSON" | python3 -c '
import json, sys
data = json.load(sys.stdin)
for f in data["urls"]:
    if f["packagetype"] == "sdist":
        print(f["url"])
        break
')
WHEEL_URL=$(echo "$PYPI_JSON" | python3 -c '
import json, sys
data = json.load(sys.stdin)
for f in data["urls"]:
    if f["packagetype"] == "bdist_wheel":
        print(f["url"])
        break
')

if [ -z "$SDIST_URL" ] || [ -z "$WHEEL_URL" ]; then
  echo "::error::could not resolve sdist/wheel URLs from PyPI for ${PKG} ${VERSION}" >&2
  exit 1
fi

echo "Fetching sdist: $SDIST_URL"
curl -fsSL "$SDIST_URL" -o "$TMPDIR/sdist.tar.gz"
SDIST_SHA=$(sha256sum "$TMPDIR/sdist.tar.gz" | awk '{print $1}')

echo "Fetching wheel: $WHEEL_URL"
curl -fsSL "$WHEEL_URL" -o "$TMPDIR/wheel.whl"
WHEEL_SHA=$(sha256sum "$TMPDIR/wheel.whl" | awk '{print $1}')

echo "sdist sha256 = $SDIST_SHA"
echo "wheel sha256 = $WHEEL_SHA"

# --- Update Homebrew formula ---
# Homebrew prefers sdist for source builds.
python3 - "$HOMEBREW_FORMULA" "$VERSION" "$SDIST_URL" "$SDIST_SHA" <<'PY'
import re, sys
path, version, url, sha = sys.argv[1:]
with open(path) as f:
    src = f.read()
# Replace whatever value the formula's own url / sha256 / version lines hold,
# so a second run overwrites the first. Only lines indented two spaces match:
# `resource` blocks are indented four, so their url / sha256 stay as they are.
for field, value in (("url", url), ("sha256", sha), ("version", version)):
    src, count = re.subn(
        rf'^(  {field} )"[^"]*"$',
        lambda m, value=value: f'{m.group(1)}"{value}"',
        src,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        sys.exit(f"{path}: found no top-level `{field} \"...\"` line to update")
with open(path, "w") as f:
    f.write(src)
PY

# --- Update Scoop manifest ---
python3 - "$SCOOP_MANIFEST" "$VERSION" "$WHEEL_URL" "$WHEEL_SHA" <<'PY'
import json, sys
path, version, url, sha = sys.argv[1:]
with open(path) as f:
    data = json.load(f)
data["version"] = version
data["url"] = url
data["hash"] = sha
with open(path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PY

echo ""
echo "Updated:"
echo "  $HOMEBREW_FORMULA"
echo "  $SCOOP_MANIFEST"

# Homebrew installs the package with `pip --no-deps`, so the formula must
# list every dependency as a `resource` block.
if ! grep -Eq '^[[:space:]]*resource[[:space:]]+"[^"]+"[[:space:]]+do' "$HOMEBREW_FORMULA"; then
  echo "" >&2
  echo "WARNING: $HOMEBREW_FORMULA has no resource blocks." >&2
  echo "  Without them, brew install leaves out pydantic, websockets, httpx," >&2
  echo "  keyring and the rest, and autogpt-shim fails to start." >&2
  echo "  Do step 2 below before you copy the formula to the tap." >&2
fi

echo ""
echo "Next:"
echo "  1. Review the diffs."
echo "  2. Generate the formula's resource blocks (needs Homebrew). Do this"
echo "     on every release, because the dependencies change:"
echo "       brew update-python-resources $HOMEBREW_FORMULA"
echo "  3. Copy the files into the tap / bucket repos:"
echo "     - Significant-Gravitas/homebrew-tap → Formula/${PKG}.rb"
echo "     - Significant-Gravitas/scoop-bucket → bucket/${PKG}.json"
echo "  4. Open PRs against those repos."
