# MrScraper Status
<!-- MANUAL: file_description -->
Reads your MrScraper account and plan. Needs a MrScraper API key: create one at [app.mrscraper.com/api-tokens](https://app.mrscraper.com/api-tokens) and add it as a MrScraper credential. MrScraper is also available as a hosted MCP server under Integrations.
<!-- END MANUAL -->

## MrScraper Status

### What it is
Return the raw account status API response without summarizing, redacting body fields, or adding calculated fields.

### How it works
<!-- MANUAL: how_it_works -->
The block sends `GET https://api.app.mrscraper.com/api/v1/subscription-accounts` with the API key in the `x-api-token` header and outputs the body unchanged as `response`. `account` is the body's `data` field: the token limit and usage, the plan (`subscriptionItem`) and the remaining scraper allowance (`scraperAllowance`). The call has a fixed 600-second deadline.
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| response | Raw account status API response, with message and data. | Dict[str, Any] |
| account | Account details from the response's data field, including subscriptionItem and scraperAllowance. | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Check the balance before a large run**: Read `account.scraperAllowance.remaining` and stop a workflow before it starts a batch of scrapes the plan cannot cover.

**Usage reporting**: Record `tokenUsage` and `tokenLimit` on a schedule to track spend over time.
<!-- END MANUAL -->

---
