# AgentMail Threads
<!-- MANUAL: file_description -->
Blocks for listing, retrieving, and deleting conversation threads in AgentMail. Threads group related messages into a single conversation and can be queried per-inbox or across the entire organization.
<!-- END MANUAL -->

## List Inbox Threads

### What it is
A block that lists all conversation threads within a specific AgentMail inbox.

### How it works
<!-- MANUAL: how_it_works -->
The block retrieves a paginated list of threads from the specified inbox. Use labels to filter threads by campaign, status, or custom tags. Supports pagination via page tokens for inboxes with many conversations.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address to list threads from | str | Yes |
| limit | Maximum number of threads to return per page (1-100) | int | No |
| page_token | Token from a previous response to fetch the next page | str | No |
| labels | Only return threads matching ALL of these labels | List[str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| threads | List of thread objects with thread_id, subject, message count, labels, etc. | List[dict] |
| count | Number of threads returned | int |
| next_page_token | Token for the next page. Empty if no more results. | str |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Inbox Dashboard**: Build a view of all active conversations in an agent's inbox, filtered by status labels.

**Campaign Tracking**: List all threads tagged with a specific campaign label to review conversation progress.
<!-- END MANUAL -->

---

## Get Inbox Thread

### What it is
A block that retrieves a single conversation thread with all its messages from an AgentMail inbox.

### How it works
<!-- MANUAL: how_it_works -->
The block fetches the thread and returns all messages in chronological order. Use this to get the full conversation history before composing a reply, providing complete context to an AI agent.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address the thread belongs to | str | Yes |
| thread_id | Thread ID to retrieve | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| thread_id | Unique identifier of the thread | str |
| messages | All messages in the thread, in chronological order | List[dict] |
| result | Complete thread object with all metadata | dict |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Contextual Replies**: Fetch the full conversation history so an AI agent can generate an informed, context-aware reply.

**Conversation Analysis**: Retrieve all messages in a thread for sentiment analysis or topic extraction.
<!-- END MANUAL -->

---

## Delete Inbox Thread

### What it is
A block that permanently deletes a conversation thread and all its messages from an inbox.

### How it works
<!-- MANUAL: how_it_works -->
The block removes the thread and every message within it from the specified inbox. This action cannot be undone. Use with caution — consider archiving or labeling threads instead if you may need them later.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address the thread belongs to | str | Yes |
| thread_id | Thread ID to permanently delete | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| success | True if the thread was successfully deleted | bool |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Data Cleanup**: Remove completed or spam conversations to keep the inbox organized.

**Privacy Compliance**: Delete threads containing sensitive data after processing is complete.
<!-- END MANUAL -->

---

## List Org Threads

### What it is
A block that lists conversation threads across ALL inboxes in your AgentMail organization.

### How it works
<!-- MANUAL: how_it_works -->
Unlike per-inbox listing, this returns threads from every inbox in the organization. Supports label filtering and pagination. Ideal for supervisor agents that monitor all conversations, analytics dashboards, or cross-agent routing workflows.
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
| threads | List of thread objects from all inboxes in the organization | List[dict] |
| count | Number of threads returned | int |
| next_page_token | Token for the next page. Empty if no more results. | str |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Supervisor Agent**: Monitor all agent conversations across the organization to detect escalation-worthy threads.

**Analytics Dashboard**: Aggregate conversation data across all inboxes for reporting and insights.
<!-- END MANUAL -->

---

## Get Org Thread

### What it is
A block that retrieves a single conversation thread by ID from anywhere in the organization, without needing the inbox ID.

### How it works
<!-- MANUAL: how_it_works -->
The block fetches a thread using only its thread_id, regardless of which inbox it belongs to. Returns the thread with all its messages in chronological order. Useful when you have a thread_id but don't know or need the inbox_id.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| thread_id | Thread ID to retrieve (works across all inboxes) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| thread_id | Unique identifier of the thread | str |
| messages | All messages in the thread, in chronological order | List[dict] |
| result | Complete thread object with all metadata | dict |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Cross-Inbox Lookup**: Retrieve a thread when you only have its ID, without needing to know which inbox it belongs to.

**Webhook Processing**: When receiving a webhook notification with a thread_id, fetch the full thread for processing.
<!-- END MANUAL -->
