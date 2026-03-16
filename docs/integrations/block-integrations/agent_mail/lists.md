# Agent Mail Lists
<!-- MANUAL: file_description -->
Blocks for managing allow/block lists in AgentMail. Lists let you control which email addresses and domains your agents can send to or receive from, based on direction (send/receive) and type (allow/block).
<!-- END MANUAL -->

## Agent Mail Create List Entry

### What it is
Add an email address or domain to an allow/block list. Block spam senders or whitelist trusted domains.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail Delete List Entry

### What it is
Remove an email address or domain from an allow/block list to stop filtering it.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail Get List Entry

### What it is
Check if an email address or domain is in an allow/block list. Verify filtering rules.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail List Entries

### What it is
List all entries in an AgentMail allow/block list. Choose send/receive direction and allow/block type.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
