# Darkmoon Findings
<!-- MANUAL: file_description -->
Blocks for working with the output of [Darkmoon](https://github.com/ASCIT31/Dark-Moon), an open source (GPL-3.0) autonomous AI penetration testing platform. These blocks only process data you already have; they make no network calls and need no credentials.
<!-- END MANUAL -->

## Darkmoon Findings Parser

### What it is
Parses the JSON findings of a Darkmoon scan (open source autonomous AI pentest engine), filters them by severity and proof status, and returns counts, a pass/fail gate and a Markdown report. Findings may include false positives and need human review.

### How it works
<!-- MANUAL: how_it_works -->
The block decodes the findings JSON of a Darkmoon scan: either a bare array of findings, or an object holding the array under `findings` or `data`. Each finding keeps all of its fields (title, severity, cvss_score, category, status, description, endpoint, cve, remediation and anything else the scan emitted). Findings are filtered by `min_severity` (`info` keeps everything, including findings with a missing or unknown severity) and, optionally, by proof status (`exploited` or `confirmed`). The kept findings are sorted most severe first. `gate_failed` is true when a kept finding reaches the `fail_on` severity, so a graph can branch on it, for example to open a ticket or stop a deployment. Input that is not a findings document raises a block error that names the problem. Findings can include false positives, so treat the gate as a triage signal and review before acting.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| findings_json | The findings JSON from a Darkmoon scan: an array of findings, or an object holding them under 'findings' or 'data'. | str | Yes |
| min_severity | Drop findings below this severity from the results. 'info' keeps everything, including findings without a recognised severity. | "info" \| "low" \| "medium" \| "high" \| "critical" | No |
| only_proven | Keep only findings whose status is 'exploited' or 'confirmed', dropping 'unconfirmed' ones. | bool | No |
| fail_on | The gate output turns on when a kept finding is at or above this severity. Choose 'never' to disable the gate. | "never" \| "low" \| "medium" \| "high" \| "critical" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| findings | The kept findings, most severe first, with all their fields. | List[Dict[str, Any]] |
| total | Number of findings kept. | int |
| severity_counts | Kept findings per severity (critical, high, medium, low, info). | Dict[str, int] |
| highest_severity | The most severe rating among kept findings, or 'none'. | str |
| gate_failed | True when a kept finding reaches the fail_on severity. | bool |
| markdown_report | A Markdown summary table, ready to post to chat or a ticket. | str |

### Possible use case
<!-- MANUAL: use_case -->
Feed the findings JSON of a Darkmoon scan, read from a file or an HTTP response, into this block, route `gate_failed` to a condition block that blocks a release, and post `markdown_report` to a chat or ticketing block so the team sees a severity-sorted summary.
<!-- END MANUAL -->

---
