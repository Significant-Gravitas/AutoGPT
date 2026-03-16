# AgentMail Inbox
<!-- MANUAL: file_description -->
Blocks for creating, retrieving, listing, updating, and deleting AgentMail inboxes. An Inbox is a fully programmable email account for AI agents — each inbox gets a unique email address and can send, receive, and manage emails via the AgentMail API.
<!-- END MANUAL -->

## Create Inbox

### What it is
A block that creates a new email inbox for an AI agent via AgentMail.

### How it works
<!-- MANUAL: how_it_works -->
The block creates a new inbox with a unique email address. If username and domain are not provided, AgentMail auto-generates them (e.g. random@agentmail.to). Specify a username for a recognizable address (e.g. support@agentmail.to) or use a custom domain. The display_name sets the friendly "From" name shown to recipients.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| username | Local part of the email address (e.g. 'support'). Leave empty to auto-generate. | str | No |
| domain | Email domain (e.g. 'mydomain.com'). Defaults to agentmail.to if empty. | str | No |
| display_name | Friendly name shown in the 'From' field of sent emails | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| inbox_id | Unique identifier for the created inbox | str |
| email_address | Full email address of the inbox (e.g. support@agentmail.to) | str |
| result | Complete inbox object with all metadata | dict |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Agent Fleet**: Dynamically create dedicated inboxes for each AI agent in a multi-agent system, giving each agent its own email identity.

**Customer Onboarding**: Automatically create a support inbox for each new customer or project.
<!-- END MANUAL -->

---

## Get Inbox

### What it is
A block that retrieves details of an existing AgentMail inbox by its ID or email address.

### How it works
<!-- MANUAL: how_it_works -->
The block fetches the inbox metadata including email address, display name, and configuration. Use this to check if an inbox exists or get its properties.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address to look up | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| inbox_id | Unique identifier of the inbox | str |
| email_address | Full email address of the inbox | str |
| display_name | Friendly name shown in the 'From' field | str |
| result | Complete inbox object with all metadata | dict |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Inbox Verification**: Check if an inbox exists and retrieve its configuration before sending emails from it.

**Display Configuration**: Fetch the display name and email address to show in a UI or confirmation message.
<!-- END MANUAL -->

---

## List Inboxes

### What it is
A block that lists all email inboxes in your AgentMail organization.

### How it works
<!-- MANUAL: how_it_works -->
The block retrieves a paginated list of all inboxes with their metadata. Use page_token for pagination when you have many inboxes.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| limit | Maximum number of inboxes to return per page (1-100) | int | No |
| page_token | Token from a previous response to fetch the next page | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| inboxes | List of inbox objects with inbox_id, email_address, display_name, etc. | List[dict] |
| count | Total number of inboxes in your organization | int |
| next_page_token | Token for the next page. Empty if no more results. | str |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Admin Dashboard**: Display all agent inboxes in an admin panel for management and monitoring.

**Inbox Discovery**: List available inboxes to let a user select which one to use in a workflow.
<!-- END MANUAL -->

---

## Update Inbox

### What it is
A block that updates the display name of an existing AgentMail inbox.

### How it works
<!-- MANUAL: how_it_works -->
The block changes the friendly name shown in the 'From' field when emails are sent from this inbox. The email address itself cannot be changed.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address to update | str | Yes |
| display_name | New display name for the inbox (e.g. 'Customer Support Bot') | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| inbox_id | The updated inbox ID | str |
| result | Complete updated inbox object with all metadata | dict |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Branding Updates**: Change the display name of an agent's inbox to match updated branding or team names.

**Role-Based Naming**: Update the display name based on the agent's current role or function (e.g. "Sales Agent" to "Support Agent").
<!-- END MANUAL -->

---

## Delete Inbox

### What it is
A block that permanently deletes an AgentMail inbox and all its data.

### How it works
<!-- MANUAL: how_it_works -->
The block removes the inbox, all its messages, threads, and drafts permanently. This action cannot be undone. The email address will no longer receive or send emails.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address to permanently delete | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| success | True if the inbox was successfully deleted | bool |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Cleanup**: Remove inboxes for agents that are no longer active or needed.

**Customer Offboarding**: Delete customer-specific inboxes when a customer's account is closed.
<!-- END MANUAL -->
