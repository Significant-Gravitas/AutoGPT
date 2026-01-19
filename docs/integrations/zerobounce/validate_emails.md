# Zerobounce Validate Emails
<!-- MANUAL: file_description -->
Blocks for validating email deliverability using ZeroBounce.
<!-- END MANUAL -->

## Validate Emails

### What it is
Validate emails

### How it works
<!-- MANUAL: how_it_works -->
This block uses the ZeroBounce API to validate email addresses for deliverability. It checks if an email is valid, invalid, catch-all, spamtrap, abuse, or disposable. Optionally provide an IP address for additional validation context.

The response includes detailed status information, SMTP provider, and recommendation on whether to send emails to that address.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| email | Email to validate | str | Yes |
| ip_address | IP address to validate | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the search failed | str |
| response | Response from ZeroBounce | Response |

### Possible use case
<!-- MANUAL: use_case -->
**List Cleaning**: Validate email lists before campaigns to reduce bounce rates.

**Lead Qualification**: Verify lead email addresses as part of intake workflows.

**Form Validation**: Check email validity in real-time during user registration or contact form submissions.
<!-- END MANUAL -->

---
