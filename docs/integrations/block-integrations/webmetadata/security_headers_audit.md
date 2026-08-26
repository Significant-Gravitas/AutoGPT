# Webmetadata Security Headers Audit
<!-- MANUAL: file_description -->
Blocks backed by the [Web Metadata Extractor API](https://webmetadataextractor.com/) (free tier: 1,000 requests/month, no card required).
<!-- END MANUAL -->

## Security Headers Audit

### What it is
Grades a URL's HTTP security headers (HSTS, CSP, X-Frame-Options, etc.) with a 0-100 score.

### How it works
<!-- MANUAL: how_it_works -->
The block sends the target URL to the API's `/api/v1/security` endpoint, which fetches the page once and inspects six response headers: `Strict-Transport-Security`, `Content-Security-Policy`, `X-Frame-Options`, `X-Content-Type-Options`, `Referrer-Policy`, and `Permissions-Policy`. Each header is graded independently as `missing`, `weak`, `report-only`, `reasonable`, or `strong` based on its actual directive values (e.g. a CSP with `unsafe-inline` grades weaker than one without it), and the per-header grades are combined into a single 0-100 score.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | The URL to audit | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| security_score_percentage | Overall security-headers score, 0-100 | float |
| security_header_grades | Per-header grade: missing, weak, report-only, reasonable, or strong | Dict[str, str] |
| security_headers | The raw header values that were graded | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Vendor/Competitor Due Diligence**: Have an agent check a prospective vendor's or a competitor's site for baseline security hygiene before a partnership or comparison report.

**Deploy Pipeline Gate**: Chain this block after a deploy step and branch the graph on `security_score_percentage` to alert or block a release that regresses below a minimum score.

**Bulk Portfolio Audit**: Loop this block over a list of URLs (e.g. all the sites a client manages) to flag which ones are missing HSTS or CSP and need remediation.
<!-- END MANUAL -->

---
