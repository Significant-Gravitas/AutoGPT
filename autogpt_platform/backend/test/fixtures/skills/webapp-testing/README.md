# `webapp-testing` — vendored test fixture

Verbatim copy of `skills/webapp-testing` from
[anthropics/skills](https://github.com/anthropics/skills) at commit
`34040c9c56`, used as the real-world package the marketplace publish/install
tests round-trip. Six files, 22,394 bytes, one `100755` script — the shape a
hand-written fixture would not have.

Licensed Apache-2.0 (`LICENSE.txt`, Copyright 2026 Anthropic, PBC). The other
packages in that repository (`pdf`, `docx`, `pptx`, `xlsx`) are proprietary and
are deliberately not vendored.

Do not edit: the tests assert byte-equality against what publish stored, and
`scripts/with_server.py` must keep its executable bit.
