# Agent Mail Threads
<!-- MANUAL: file_description -->
Blocks for listing, retrieving, and deleting conversation threads in AgentMail. Threads group related messages into a single conversation and can be queried per-inbox or across the entire organization.
<!-- END MANUAL -->

## Agent Mail Delete Inbox Thread

### What it is
Permanently delete a conversation thread and all its messages. This action cannot be undone.

### How it works
<!-- MANUAL: how_it_works -->
The block calls the AgentMail API to permanently delete a thread and all of its messages from the specified inbox. It requires both the inbox ID (or email address) and the thread ID.

On success the block outputs `success=True`. If the API returns an error (for example, the thread does not exist), the error propagates to the global error handler and the block outputs an error message instead.
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
- **GDPR Data Removal** — Permanently delete conversation threads when a user requests erasure of their personal data.
- **Spam Cleanup** — Automatically remove threads flagged as spam by an upstream classification block.
- **Conversation Archival Pipeline** — Delete threads from the live inbox after they have been exported to long-term storage.
<!-- END MANUAL -->

---

## Agent Mail Get Inbox Thread

### What it is
Retrieve a conversation thread with all its messages. Use for getting full conversation context before replying.

### How it works
<!-- MANUAL: how_it_works -->
The block fetches a single thread from a specific inbox by calling the AgentMail API with the inbox ID and thread ID. It returns the thread ID, the full list of messages in chronological order, and the complete thread object as a dictionary.

Any API error (such as an invalid thread ID or insufficient permissions) propagates to the global error handler, and the block outputs an error message.
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
- **Context-Aware Replies** — Retrieve the full conversation history before generating an AI-drafted reply to ensure continuity.
- **Conversation Summarization** — Pull all messages in a thread and pass them to a summarization block for a digest.
- **Support Ticket Review** — Fetch a specific customer thread so a QA agent can evaluate response quality.
<!-- END MANUAL -->

---

## Agent Mail Get Org Thread

### What it is
Retrieve a conversation thread by ID from anywhere in the organization, without needing the inbox ID.

### How it works
<!-- MANUAL: how_it_works -->
The block performs an organization-wide thread lookup by calling the AgentMail API with only the thread ID. Unlike the inbox-scoped variant, no inbox ID is required because the API resolves the thread across all inboxes in the organization.

It returns the thread ID, all messages in chronological order, and the complete thread object. Errors propagate to the global error handler.
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
- **Cross-Inbox Thread Tracking** — Look up a thread by ID when the originating inbox is unknown, such as from a webhook or external reference.
- **Supervisor Agent Oversight** — Allow a manager agent to inspect any conversation across the organization without needing inbox-level routing.
- **Audit and Compliance** — Retrieve a specific thread for compliance review when only the thread ID is available from a log or report.
<!-- END MANUAL -->

---

## Agent Mail List Inbox Threads

### What it is
List all conversation threads in an AgentMail inbox. Filter by labels for campaign tracking or status management.

### How it works
<!-- MANUAL: how_it_works -->
The block lists conversation threads within a single inbox by calling the AgentMail API with the inbox ID and optional pagination and filtering parameters. You can set a limit (1-100), pass a page token for pagination, and filter by labels so that only threads matching all specified labels are returned.

The block outputs the list of thread objects, the count of threads returned in this page, and a next-page token for retrieving additional results. Errors propagate to the global error handler.
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
- **Inbox Dashboard** — List all threads in a support inbox to display an overview of active conversations.
- **Campaign Monitoring** — Filter threads by a campaign label to track how many conversations a specific outreach effort has generated.
- **Stale Thread Detection** — Paginate through all threads in an inbox to identify conversations that have not received a reply within a set time window.
<!-- END MANUAL -->

---

## Agent Mail List Org Threads

### What it is
List threads across ALL inboxes in your organization. Use for supervisor agents, dashboards, or cross-agent monitoring.

### How it works
<!-- MANUAL: how_it_works -->
The block lists threads across all inboxes in the organization by calling the AgentMail API without an inbox ID. It accepts optional limit, page-token, and label-filter parameters, which are forwarded directly to the API.

Results include threads from every inbox the organization owns. The block outputs the list of thread objects, the count for the current page, and a next-page token for pagination. Errors propagate to the global error handler.
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
- **Organization-Wide Activity Feed** — Build a real-time dashboard showing the latest conversations across every agent inbox.
- **Cross-Agent Analytics** — Aggregate thread counts and labels across all inboxes to measure overall communication volume and topic distribution.
- **Escalation Routing** — Scan all org threads for a specific label (e.g., "urgent") and route matching threads to a dedicated escalation agent.
<!-- END MANUAL -->

---
