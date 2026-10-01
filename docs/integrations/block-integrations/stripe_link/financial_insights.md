# Stripe Link Financial Insights
<!-- MANUAL: file_description -->
These read-only blocks answer questions about the bank accounts and credit
cards a user has shared through Stripe Link: which accounts there are, their
balances, and their transactions. Nothing here moves money. Link offers
financial insights to US consumers only, and only on live accounts; there is
no sandbox.

The user chooses which accounts to share when they connect Stripe Link, and can
revoke an account in the Link app at any time. A connection made before the
platform asked for this access, or one whose accounts were revoked, gets an
error asking the user to reconnect. A deployment that connects Stripe Link
through its own OAuth client (`STRIPE_LINK_CLIENT_ID` and the related settings)
must also register that Stripe account for Financial Connections in the Stripe
Dashboard; without the registration Stripe allows only Link's own transactions
to be requested.
<!-- END MANUAL -->

## Stripe Link Get Balances

### What it is
Get current balances for the bank accounts and credit cards the user shared through Stripe Link: the posted balance, plus available funds for bank accounts or credit used for cards. Amounts are integers in the smallest currency unit (cents for USD).

### How it works
<!-- MANUAL: how_it_works -->
The block makes an authenticated `GET /balances` request to Link, adding one
`sources[]` parameter per account when `source_ids` is set, and follows
`has_more` until every balance is read. Each balance keeps Link's posted
`current` amount and its currency; `cash.available` becomes `available` for
bank accounts and `credit.used` becomes `used` for cards, both keyed by
currency, such as `{"usd": 31005}`. Every amount stays in the currency's
smallest unit.

Right after an account is connected, Link may answer 202
`external_data_retrieval_pending`. The block waits and retries three times,
about 14 seconds in all, then asks the user to try again shortly. A 401 or 403
means the grant does not cover the data, so the block asks the user to
reconnect Stripe Link instead of retrying.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| source_ids | Only these accounts, by ID from List Financial Accounts. Leave empty for every account the user shared. | List[str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| balances | One balance per account, in the smallest unit of that account's currency (cents for USD). Never add balances in different currencies together. | List[LinkBalance] |

### Possible use case
<!-- MANUAL: use_case -->
**Cash Position**: Report how much money is available across the user's checking and savings accounts.

**Card Utilization**: Show how much credit is in use on each connected card before a large purchase.

**Low-Balance Alerts**: Check balances on a schedule and notify the user when an account drops below a threshold they chose.
<!-- END MANUAL -->

---

## Stripe Link List Financial Accounts

### What it is
List the bank accounts and credit cards the user shared through Stripe Link for financial insights, with each one's ID, name, last four digits and whether its balances and transactions are ready to read. Use the IDs to narrow List Transactions or Get Balances to particular accounts. To pay with Link, use List Payment Methods instead.

### How it works
<!-- MANUAL: how_it_works -->
The block makes an authenticated `GET /sources` request to Link and follows
`has_more` until every shared account is read. Each account is reduced to its
ID, name, type, last four digits, card brand or bank name, connection status,
granted actions, and a `capabilities` map such as
`{"balances": "eligible", "transactions": "pending"}`. Anything else Link
returns is dropped, so new account details cannot reach persisted outputs
without review.

A capability marked `pending` is still loading after the user connected the
account; `eligible` data can be read now. Refused access (401 or 403) produces
an error asking the user to reconnect Stripe Link and share their accounts.
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| accounts | Every bank account and card the user shared, with what each one is ready to provide | List[LinkFinancialAccount] |

### Possible use case
<!-- MANUAL: use_case -->
**Account Picker**: Let the user choose which accounts a budgeting agent should analyze, then pass their IDs to the other blocks.

**Readiness Check**: Confirm an account's balances and transactions are `eligible` before building a report from them.

**Sharing Review**: Show the user exactly which accounts and kinds of data they have shared with the agent.
<!-- END MANUAL -->

---

## Stripe Link List Transactions

### What it is
Read the user's transactions from the bank accounts and credit cards they shared through Stripe Link, filtered by date range, account or origin. Use it for questions about spending, income, merchants or subscriptions, such as how much they spent on dining last month. Amounts are integers in the smallest currency unit, negative for money leaving the account. When has_more is true, pass next_cursor back as starting_after to keep reading.

### How it works
<!-- MANUAL: how_it_works -->
The block makes authenticated `GET /transactions` requests to Link with the
chosen filters: the dates go out as `date_start` and `date_end`, each account
as a repeated `sources[]` parameter, and `origin` and `category` only when set.
Link returns at most 100 transactions per request, so for a larger `limit` the
block requests further pages with the same filters, moving `starting_after` to
the last transaction received, up to 500 per run. When more remain, `has_more`
is true and `next_cursor` holds the ID to continue from.

Dates must be real calendar dates written YYYY-MM-DD, with the start no later
than the end. Statuses pass through exactly as Link reports them, including
values the block does not know. A transaction without an ID, amount, currency
or date fails the run rather than being skipped, because a total that silently
left it out would be wrong. Data that is still loading and refused access are
handled as in Get Balances.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| start_date | Only transactions on or after this date, written YYYY-MM-DD. Leave empty for no lower bound. | str | No |
| end_date | Only transactions on or before this date, written YYYY-MM-DD. Leave empty for no upper bound. | str | No |
| source_ids | Only these accounts, by ID from List Financial Accounts. Leave empty for every account the user shared. | List[str] | No |
| origin | `external_connection` for transactions from the user's connected banks and cards, `link` for purchases made through Link, `all` for both. | "all" \| "external_connection" \| "link" | No |
| category | Only this spending category. Many transactions have no category, so leave this empty when a total must be complete. | str | No |
| limit | Most transactions to return, 1 to 500. Above 100 the block reads further pages from Link itself. | int | No |
| starting_after | Continue after this transaction ID: pass `next_cursor` from the previous run and keep every other input the same. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| transactions | Matching transactions in the order Link returns them. Amounts are integers in the currency's smallest unit (cents for USD), negative for money leaving the account. | List[LinkTransaction] |
| has_more | True when more matching transactions exist beyond these. A total over a date range is incomplete until this is false. | bool |
| next_cursor | Pass as `starting_after`, with the same filters, to read the next transactions. Empty when there are no more. | str |

### Possible use case
<!-- MANUAL: use_case -->
**Monthly Spending Summary**: Total last month's spending by category or merchant for a budgeting report.

**Subscription Finder**: Spot recurring charges from the same merchant across several months.

**Expense Export**: Pull a date range of card transactions into a spreadsheet for bookkeeping.
<!-- END MANUAL -->

---
