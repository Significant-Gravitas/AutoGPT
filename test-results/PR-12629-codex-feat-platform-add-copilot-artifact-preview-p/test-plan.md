# Test Plan: PR #12629 — AutoPilot Artifacts Panel

## Scenarios
1. Artifact panel opens when agent creates a file — verify card appears, panel slides in
2. Panel header actions — copy, download, maximize/restore, minimize/expand, close
3. Source/Preview toggle — for markdown/HTML/JSON files
4. CSV renders as sortable table
5. HTML renders in sandboxed iframe
6. JSON renders as interactive tree
7. Code renders with syntax highlighting (not markdown)
8. Image renders directly in panel
9. PDF renders via embed
10. Drag handle resizes panel
11. Escape key closes panel

## Negative Tests
1. Binary file (if any) shows download-only card, not openable
2. Panel close restores chat to centered layout
