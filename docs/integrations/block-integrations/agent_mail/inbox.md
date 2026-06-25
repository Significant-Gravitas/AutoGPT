# Agent Mail Inbox
<!-- MANUAL: file_description -->
Blocks for creating, retrieving, listing, updating, and deleting AgentMail inboxes. An Inbox is a fully programmable email account for AI agents — each inbox gets a unique email address and can send, receive, and manage emails via the AgentMail API.
<!-- END MANUAL -->

## Agent Mail Create Inbox

### What it is
Create a new email inbox for an AI agent via AgentMail. Each inbox gets a unique address and can send/receive emails.

### How it works
<!-- MANUAL: how_it_works -->
This block calls the AgentMail API to provision a new inbox. You can optionally specify a username (local part of the address), a custom domain, and a display name. Any parameters left empty use sensible defaults — the username is auto-generated and the domain defaults to `agentmail.to`.

The API returns the newly created inbox object, from which the block extracts the inbox ID and full email address as separate outputs. The complete inbox metadata is also available as a dictionary for downstream blocks that need additional fields. If the API call fails, the error propagates to the global error handler.
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
**Per-Customer Support Agents** — Spin up a dedicated inbox for each customer so an AI agent can handle their support requests through a personalized email address.

**Campaign-Specific Outreach** — Create a branded inbox for each marketing campaign so replies are automatically routed to the agent managing that campaign.

**Ephemeral Verification Workflows** — Generate a temporary inbox on the fly to receive a sign-up confirmation code, then pass the address to a registration block.
<!-- END MANUAL -->

---

## Agent Mail Delete Inbox

### What it is
Permanently delete an AgentMail inbox and all its messages, threads, and drafts. This action cannot be undone.

### How it works
<!-- MANUAL: how_it_works -->
This block sends a delete request to the AgentMail API using the provided inbox ID. The API permanently removes the inbox along with all associated messages, threads, and drafts. This action is irreversible.

On success the block outputs `success=True`. If the API returns an error (e.g., the inbox does not exist), the exception propagates to the global error handler.
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
**Cleanup After Task Completion** — Delete a temporary inbox once an agent finishes processing a one-off workflow like order confirmation or password reset.

**User Offboarding** — Remove an agent's inbox when a customer cancels their account to avoid accumulating unused resources.

**Test Environment Teardown** — Automatically delete inboxes created during integration tests so the organization stays clean between runs.
<!-- END MANUAL -->

---

## Agent Mail Get Inbox

### What it is
Retrieve details of an existing AgentMail inbox including its email address, display name, and configuration.

### How it works
<!-- MANUAL: how_it_works -->
This block calls the AgentMail API to retrieve the details of a single inbox identified by its inbox ID (which is also its email address). The API returns the full inbox object including its email address, display name, and any other metadata.

The block extracts the inbox ID, email address, and display name as individual outputs for easy wiring to downstream blocks. The complete inbox object is also returned as a dictionary. If the inbox does not exist or the request fails, the error propagates to the global error handler.
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
**Pre-Send Validation** — Fetch inbox details before sending an email to confirm the inbox still exists and verify its display name is correct.

**Dashboard Display** — Retrieve inbox metadata to show the agent's email address and display name in a monitoring dashboard.

**Conditional Routing** — Look up an inbox's properties to decide which downstream workflow should handle incoming messages for that address.
<!-- END MANUAL -->

---

## Agent Mail List Inboxes

### What it is
List all email inboxes in your AgentMail organization with pagination support.

### How it works
<!-- MANUAL: how_it_works -->
This block calls the AgentMail API to list all inboxes in the organization. You can control page size with the `limit` parameter (1-100) and paginate through results using the `page_token` returned from a previous call. Only non-empty parameters are sent in the request.

The block outputs the list of inbox objects, a count of inboxes returned, and a `next_page_token` for fetching additional pages. When there are no more results, `next_page_token` is empty. If the API call fails, the error propagates to the global error handler.
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
**Organization Audit** — Enumerate all inboxes to generate a report of active agent email accounts and their configurations.

**Bulk Operations** — Iterate through every inbox to perform batch updates, such as rotating API keys or updating display names across all agents.

**Stale Inbox Detection** — List all inboxes and cross-reference with recent activity to identify accounts that can be cleaned up.
<!-- END MANUAL -->

---

## Agent Mail Update Inbox

### What it is
Update the display name of an AgentMail inbox. Changes the 'From' name shown when emails are sent.

### How it works
<!-- MANUAL: how_it_works -->
This block calls the AgentMail API to update an existing inbox's display name. It sends the inbox ID and the new display name to the update endpoint. The display name controls the "From" label recipients see when the agent sends emails.

The block outputs the inbox ID and the full updated inbox object as a dictionary. If the inbox does not exist or the request fails, the error propagates to the global error handler.
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
**Rebranding an Agent** — Change the display name when an agent is reassigned from one team to another so outgoing emails reflect the new identity.

**Personalized Sender Names** — Update the display name dynamically to include a customer's name or ticket number for a more personal touch in automated replies.

**A/B Testing Email Identity** — Swap the display name between variations to test which sender identity gets higher open rates.
<!-- END MANUAL -->

---
