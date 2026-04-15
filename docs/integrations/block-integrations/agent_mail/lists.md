# Agent Mail Lists
<!-- MANUAL: file_description -->
Blocks for managing allow/block lists in AgentMail. Lists let you control which email addresses and domains your agents can send to or receive from, based on direction (send/receive) and type (allow/block).
<!-- END MANUAL -->

## Agent Mail Create List Entry

### What it is
Add an email address or domain to an allow/block list. Block spam senders or whitelist trusted domains.

### How it works
<!-- MANUAL: how_it_works -->
This block calls the AgentMail API to add an email address or domain to a specified list. You select a direction ("send" or "receive") and a list type ("allow" or "block"), then provide the entry to add. For block lists, you can optionally include a reason such as "spam" or "competitor."

The API creates the entry and returns both the entry string and the complete entry object with metadata. Any errors propagate directly to the global error handler.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| direction | 'send' for outgoing email rules, 'receive' for incoming email rules | "send" \| "receive" | Yes |
| list_type | 'allow' to whitelist, 'block' to blacklist | "allow" \| "block" | Yes |
| entry | Email address (user@example.com) or domain (example.com) to add | str | Yes |
| reason | Reason for blocking (only used with block lists, e.g. 'spam', 'competitor') | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| entry | The email address or domain that was added | str |
| result | Complete entry object | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Spam Prevention** — Block a known spam domain so your agent never processes incoming messages from it.

**Trusted Partner Allowlisting** — Add a partner's domain to the receive allow list so their emails always reach your agent.

**Outbound Restriction** — Add a competitor's domain to the send block list to prevent your agent from accidentally emailing them.
<!-- END MANUAL -->

---

## Agent Mail Delete List Entry

### What it is
Remove an email address or domain from an allow/block list to stop filtering it.

### How it works
<!-- MANUAL: how_it_works -->
This block calls the AgentMail API to remove an existing entry from an allow or block list. You specify the direction, list type, and the entry to delete.

On success the block returns `success=True`. If the entry does not exist or the API call fails, the error propagates to the global error handler.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| direction | 'send' for outgoing rules, 'receive' for incoming rules | "send" \| "receive" | Yes |
| list_type | 'allow' for whitelist, 'block' for blacklist | "allow" \| "block" | Yes |
| entry | Email address or domain to remove from the list | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | True if the entry was successfully removed | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Unblocking a Sender** — Remove a previously blocked domain after confirming it is no longer a spam source.

**Revoking Access** — Delete a domain from the receive allow list when a partnership ends and messages should no longer be accepted.

**Policy Correction** — Remove an entry that was added by mistake so normal email flow resumes immediately.
<!-- END MANUAL -->

---

## Agent Mail Get List Entry

### What it is
Check if an email address or domain is in an allow/block list. Verify filtering rules.

### How it works
<!-- MANUAL: how_it_works -->
This block queries the AgentMail API to check whether a specific email address or domain exists in an allow or block list. You provide the direction, list type, and the entry to look up.

If the entry is found, the block returns the entry string and the complete entry object with metadata (such as the reason it was added). If the entry does not exist, the error propagates to the global error handler.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| direction | 'send' for outgoing rules, 'receive' for incoming rules | "send" \| "receive" | Yes |
| list_type | 'allow' for whitelist, 'block' for blacklist | "allow" \| "block" | Yes |
| entry | Email address or domain to look up | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| entry | The email address or domain that was found | str |
| result | Complete entry object with metadata | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Pre-Send Validation** — Check whether a recipient domain is on the send block list before your agent drafts an outbound email.

**Audit Verification** — Look up a specific address to confirm it was added to the block list and review the recorded reason.

**Conditional Routing** — Check the receive allow list for a sender's domain to decide whether to process the message or skip it.
<!-- END MANUAL -->

---

## Agent Mail List Entries

### What it is
List all entries in an AgentMail allow/block list. Choose send/receive direction and allow/block type.

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves all entries from a specified allow or block list by calling the AgentMail API. You select a direction and list type, and optionally set a page size with `limit` and continue paginating with `page_token`.

The API returns the list of entry objects, a count of entries in the current page, and a `next_page_token` for fetching additional results. Any errors propagate to the global error handler.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| direction | 'send' to filter outgoing emails, 'receive' to filter incoming emails | "send" \| "receive" | Yes |
| list_type | 'allow' for whitelist (only permit these), 'block' for blacklist (reject these) | "allow" \| "block" | Yes |
| limit | Maximum number of entries to return per page | int | No |
| page_token | Token from a previous response to fetch the next page | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| entries | List of entries, each with an email address or domain | List[Dict[str, Any]] |
| count | Number of entries returned | int |
| next_page_token | Token for the next page. Empty if no more results. | str |

### Possible use case
<!-- MANUAL: use_case -->
**Policy Dashboard** — Fetch all entries on the receive block list to display a management view where administrators can review and remove entries.

**Periodic Cleanup** — Page through the full send allow list to identify stale entries that no longer correspond to active partners.

**Compliance Export** — Retrieve every block list entry across both directions to generate a report for an internal compliance audit.
<!-- END MANUAL -->

---
