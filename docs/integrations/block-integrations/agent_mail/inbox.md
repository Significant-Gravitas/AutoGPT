# Agent Mail Inbox
<!-- MANUAL: file_description -->
Blocks for creating, retrieving, listing, updating, and deleting AgentMail inboxes. An Inbox is a fully programmable email account for AI agents — each inbox gets a unique email address and can send, receive, and manage emails via the AgentMail API.
<!-- END MANUAL -->

## Agent Mail Create Inbox

### What it is
Create a new email inbox for an AI agent via AgentMail. Each inbox gets a unique address and can send/receive emails.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| username | Local part of the email address (e.g. 'support' for support@domain.com). Leave empty to auto-generate. | str | No |
| domain | Email domain (e.g. 'mydomain.com'). Defaults to agentmail.to if empty. | str | No |
| display_name | Friendly name shown in the 'From' field of sent emails (e.g. 'Support Agent') | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| inbox_id | Unique identifier for the created inbox (also the email address) | str |
| email_address | Full email address of the inbox (e.g. support@agentmail.to) | str |
| result | Complete inbox object with all metadata | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail Delete Inbox

### What it is
Permanently delete an AgentMail inbox and all its messages, threads, and drafts. This action cannot be undone.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address to permanently delete | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | True if the inbox was successfully deleted | bool |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail Get Inbox

### What it is
Retrieve details of an existing AgentMail inbox including its email address, display name, and configuration.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address to look up (e.g. 'support@agentmail.to') | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| inbox_id | Unique identifier of the inbox | str |
| email_address | Full email address of the inbox | str |
| display_name | Friendly name shown in the 'From' field | str |
| result | Complete inbox object with all metadata | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail List Inboxes

### What it is
List all email inboxes in your AgentMail organization with pagination support.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| limit | Maximum number of inboxes to return per page (1-100) | int | No |
| page_token | Token from a previous response to fetch the next page of results | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| inboxes | List of inbox objects, each containing inbox_id, email_address, display_name, etc. | List[Dict[str, Any]] |
| count | Total number of inboxes in your organization | int |
| next_page_token | Token to pass as page_token to get the next page. Empty if no more results. | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail Update Inbox

### What it is
Update the display name of an AgentMail inbox. Changes the 'From' name shown when emails are sent.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address to update (e.g. 'support@agentmail.to') | str | Yes |
| display_name | New display name for the inbox (e.g. 'Customer Support Bot') | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| inbox_id | The updated inbox ID | str |
| result | Complete updated inbox object with all metadata | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
