# AgentMail Lists
<!-- MANUAL: file_description -->
Blocks for managing allow/block lists in AgentMail. Lists let you control which email addresses and domains your agents can send to or receive from, based on direction (send/receive) and type (allow/block).
<!-- END MANUAL -->

## List Entries

### What it is
A block that lists all entries in an AgentMail allow/block list.

### How it works
<!-- MANUAL: how_it_works -->
The block retrieves email addresses and domains that are currently in a specific list. Choose the direction ('send' or 'receive') and list type ('allow' or 'block') to query one of the four lists:
- **receive + allow**: Only accept emails from these addresses/domains
- **receive + block**: Reject emails from these addresses/domains
- **send + allow**: Only send emails to these addresses/domains
- **send + block**: Prevent sending emails to these addresses/domains
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| direction | 'send' to filter outgoing emails, 'receive' to filter incoming emails | "send" \| "receive" | Yes |
| list_type | 'allow' for whitelist, 'block' for blacklist | "allow" \| "block" | Yes |
| limit | Maximum number of entries to return per page | int | No |
| page_token | Token from a previous response to fetch the next page | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| entries | List of entries, each with an email address or domain | List[dict] |
| count | Number of entries returned | int |
| next_page_token | Token for the next page. Empty if no more results. | str |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Audit Filtering Rules**: Review which addresses or domains are currently blocked or allowed to ensure your email filtering is correct.

**Compliance Check**: List all blocked senders to verify that known spam or malicious domains are properly filtered.
<!-- END MANUAL -->

---

## Create List Entry

### What it is
A block that adds an email address or domain to an AgentMail allow/block list.

### How it works
<!-- MANUAL: how_it_works -->
The block adds an entry to the specified list. Entries can be full email addresses (e.g. 'partner@example.com') or entire domains (e.g. 'example.com'). For block lists, you can optionally provide a reason (e.g. 'spam', 'competitor') for documentation purposes.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| direction | 'send' for outgoing email rules, 'receive' for incoming email rules | "send" \| "receive" | Yes |
| list_type | 'allow' to whitelist, 'block' to blacklist | "allow" \| "block" | Yes |
| entry | Email address (user@example.com) or domain (example.com) to add | str | Yes |
| reason | Reason for blocking (only used with block lists) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| entry | The email address or domain that was added | str |
| result | Complete entry object | dict |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Spam Prevention**: Block domains known for sending spam to keep agent inboxes clean.

**Trusted Sender Whitelist**: Allow only emails from verified partner domains to reach your agent inbox.
<!-- END MANUAL -->

---

## Get List Entry

### What it is
A block that checks if an email address or domain exists in an AgentMail allow/block list.

### How it works
<!-- MANUAL: how_it_works -->
The block looks up a specific address or domain in the specified list and returns the entry details if found. Use this to verify whether a specific address or domain is currently allowed or blocked.
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
| entry | The email address or domain that was found | str |
| result | Complete entry object with metadata | dict |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Pre-Send Check**: Verify that a recipient domain is not on the block list before attempting to send an email.

**Rule Verification**: Check if a specific address has already been added to a list before creating a duplicate entry.
<!-- END MANUAL -->

---

## Delete List Entry

### What it is
A block that removes an email address or domain from an AgentMail allow/block list.

### How it works
<!-- MANUAL: how_it_works -->
The block removes the specified entry from the list. After removal, the address or domain will no longer be filtered by this list.
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
| success | True if the entry was successfully removed | bool |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Unblock Sender**: Remove a previously blocked address or domain when it's no longer considered a threat.

**Policy Update**: Remove an allow list entry when a partnership ends and emails should no longer be whitelisted.
<!-- END MANUAL -->
