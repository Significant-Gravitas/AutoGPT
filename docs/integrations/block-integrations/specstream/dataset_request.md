# Specstream Dataset Request
<!-- MANUAL: file_description -->
Blocks that start a SpecStream dataset request. SpecStream searches the live web and returns rows with a source URL and read time on every value it reads from a page. A value SpecStream builds rather than reads, such as a likely email address, has no source URL. The blocks run on your own SpecStream API key, from the [SpecStream workspace](https://specstream.ai/workspace?utm_source=autogpt).
<!-- END MANUAL -->

## SpecStream Create Dataset Request

### What it is
Reads a dataset request in plain words and returns a signed quote with the most it can cost. Nothing runs and nothing is charged.

### How it works
<!-- MANUAL: how_it_works -->
The block sends the request to SpecStream, which reads it into conditions every row must meet and details to bring back. It then asks for a signed quote for the number of rows the request names, or the row_count input when it names none. Nothing runs and nothing is charged. If SpecStream cannot read the request as a runnable dataset, the block fails with the reason so the request can be reworded. If the SpecStream service is down, the block says so instead.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| request | The dataset to build, in plain words: what each row is, the conditions every row must meet, and the details to bring back. | str | Yes |
| row_count | How many rows to ask for when the request names no number. A number in the request, such as '100 fintech startups', wins. The price is set per matching row. | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| reading | How SpecStream read the request, as one sentence | str |
| notices | Notes to show a person before the quote is approved | List[str] |
| quote_id | The ID of the signed quote | str |
| quote_hash | The signature of the quote, needed to approve it | str |
| maximum_cents | The most this run can cost, in cents. It is never exceeded. | int |
| currency | The currency of the quote | str |
| expires_at | When the quote stops being valid | str |
| price_summary | The quote in one sentence, for a person to review | str |

### Possible use case
<!-- MANUAL: use_case -->
**Grant Search**: Describe grants open to nonprofits in Texas and get a quote for the rows with the deadline and the funder.

**Lead List Pricing**: Ask for a list of companies that match your buyer profile and see the most it can cost before anything runs.

**Approval Gate**: Pass the quote to a person for review, so nothing is charged until they confirm the maximum.
<!-- END MANUAL -->

---
