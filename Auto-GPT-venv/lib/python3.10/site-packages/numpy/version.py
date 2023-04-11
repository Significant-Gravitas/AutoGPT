from __future__ import annotations

from ._version import get_versions

__ALL__ = ['version', '__version__', 'full_version', 'git_revision', 'release']

vinfo: dict[str, str] = get_versions()
version = vinfo["version"]
__version__ = vinfo.get("closest-tag", vinfo["version"])
full_version = vinfo['version']
git_revision = vinfo['full-revisionid']
release = 'dev0' not in version and '+' not in version
short_version = vinfo['version'].split("+")[0]

del get_versions, vinfo
