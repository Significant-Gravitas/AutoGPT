# Agent Mail Threads
<!-- MANUAL: file_description -->
Blocks for listing, retrieving, and deleting conversation threads in AgentMail. Threads group related messages into a single conversation and can be queried per-inbox or across the entire organization.
<!-- END MANUAL -->

## Agent Mail Delete Inbox Thread

### What it is
Permanently delete a conversation thread and all its messages. This action cannot be undone.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address the thread belongs to | str | Yes |
| thread_id | Thread ID to permanently delete | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | True if the thread was successfully deleted | bool |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail Get Inbox Thread

### What it is
Retrieve a conversation thread with all its messages. Use for getting full conversation context before replying.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address the thread belongs to | str | Yes |
| thread_id | Thread ID to retrieve | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| thread_id | Unique identifier of the thread | str |
| messages | All messages in the thread, in chronological order | List[Dict[str, Any]] |
| result | Complete thread object with all metadata | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail Get Org Thread

### What it is
Retrieve a conversation thread by ID from anywhere in the organization, without needing the inbox ID.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| thread_id | Thread ID to retrieve (works across all inboxes) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| thread_id | Unique identifier of the thread | str |
| messages | All messages in the thread, in chronological order | List[Dict[str, Any]] |
| result | Complete thread object with all metadata | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail List Inbox Threads

### What it is
List all conversation threads in an AgentMail inbox. Filter by labels for campaign tracking or status management.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address to list threads from | str | Yes |
| limit | Maximum number of threads to return per page (1-100) | int | No |
| page_token | Token from a previous response to fetch the next page | str | No |
| labels | Only return threads matching ALL of these labels (e.g. ['q4-campaign', 'follow-up']) | List[str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| threads | List of thread objects with thread_id, subject, message count, labels, etc. | List[Dict[str, Any]] |
| count | Number of threads returned | int |
| next_page_token | Token for the next page. Empty if no more results. | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail List Org Threads

### What it is
List threads across ALL inboxes in your organization. Use for supervisor agents, dashboards, or cross-agent monitoring.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| limit | Maximum number of threads to return per page (1-100) | int | No |
| page_token | Token from a previous response to fetch the next page | str | No |
| labels | Only return threads matching ALL of these labels | List[str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| threads | List of thread objects from all inboxes in the organization | List[Dict[str, Any]] |
| count | Number of threads returned | int |
| next_page_token | Token for the next page. Empty if no more results. | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
