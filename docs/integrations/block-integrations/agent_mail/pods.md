# Agent Mail Pods
<!-- MANUAL: file_description -->
Blocks for creating, managing, and querying pods in AgentMail. Pods provide multi-tenant isolation between customers — each pod acts as an isolated workspace containing its own inboxes, domains, threads, and drafts. Use pods when building SaaS platforms, agency tools, or AI agent fleets serving multiple customers.
<!-- END MANUAL -->

## Agent Mail Create Pod

### What it is
Create a new pod for multi-tenant customer isolation. Use client_id to map to your internal tenant IDs.

### How it works
<!-- MANUAL: how_it_works -->
Calls the AgentMail API to provision a new pod, passing an optional `client_id` parameter. When a `client_id` is provided, AgentMail maps it to the pod so you can later reference the pod using your own internal tenant identifier instead of the AgentMail-assigned `pod_id`.

The block returns the newly created `pod_id` along with the complete pod object containing all metadata. Any API errors propagate directly to the global error handler.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| client_id | Your internal tenant/customer ID for idempotent mapping. Lets you access the pod by your own ID instead of AgentMail's pod_id. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| pod_id | Unique identifier of the created pod | str |
| result | Complete pod object with all metadata | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
- **SaaS Customer Onboarding** — Automatically provision an isolated email workspace when a new customer signs up for your platform.
- **AI Agent Fleet Management** — Create a dedicated pod for each AI agent so its email activity is fully isolated from other agents.
- **White-Label Email Service** — Spin up tenant-scoped pods mapped to your internal customer IDs to power a branded email product.
<!-- END MANUAL -->

---

## Agent Mail Create Pod Inbox

### What it is
Create a new email inbox within a pod. The inbox is scoped to the customer workspace.

### How it works
<!-- MANUAL: how_it_works -->
Calls the AgentMail API to create a new inbox scoped to the specified pod. You can optionally provide a `username`, `domain`, and `display_name` to customize the email address and sender identity. If omitted, the username is auto-generated and the domain defaults to `agentmail.to`.

The block returns the `inbox_id`, the full `email_address`, and the complete inbox metadata object. The inbox is fully isolated within the pod, meaning it only appears in that pod's inbox listings and its threads stay separate from other pods.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| pod_id | Pod ID to create the inbox in | str | Yes |
| username | Local part of the email address (e.g. 'support'). Leave empty to auto-generate. | str | No |
| domain | Email domain (e.g. 'mydomain.com'). Defaults to agentmail.to if empty. | str | No |
| display_name | Friendly name shown in the 'From' field (e.g. 'Customer Support') | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| inbox_id | Unique identifier of the created inbox | str |
| email_address | Full email address of the inbox | str |
| result | Complete inbox object with all metadata | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
- **Per-Customer Support Addresses** — Create a `support@clientdomain.com` inbox inside each customer's pod so inbound support emails are automatically routed to the right tenant.
- **Branded Outbound Campaigns** — Provision inboxes with custom display names and domains for each customer's marketing agent to send branded emails.
- **Multi-Department Agent Setup** — Create separate inboxes (sales, billing, support) within a single customer pod so different AI agents handle different functions.
<!-- END MANUAL -->

---

## Agent Mail Delete Pod

### What it is
Permanently delete a pod. All inboxes and domains must be removed first.

### How it works
<!-- MANUAL: how_it_works -->
Calls the AgentMail API to permanently delete the specified pod. The API enforces a precondition: all inboxes and custom domains must be removed from the pod before deletion is allowed. If any remain, the API returns an error that propagates to the global error handler.

On success the block returns `success=True`. This operation is irreversible -- the pod and its associated `client_id` mapping are permanently removed.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| pod_id | Pod ID to permanently delete (must have no inboxes or domains) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | True if the pod was successfully deleted | bool |

### Possible use case
<!-- MANUAL: use_case -->
- **Customer Offboarding** — Automatically delete a customer's pod after they cancel their subscription and all their inboxes have been cleaned up.
- **Development Environment Cleanup** — Tear down temporary pods created during testing or staging so they do not accumulate over time.
- **Compliance Data Removal** — Permanently remove a tenant's email workspace as part of a GDPR or data-deletion request workflow.
<!-- END MANUAL -->

---

## Agent Mail Get Pod

### What it is
Retrieve details of an existing pod including its client_id mapping and metadata.

### How it works
<!-- MANUAL: how_it_works -->
Calls the AgentMail API with the given `pod_id` to fetch the full pod record. The returned object includes the pod's `client_id` mapping, creation timestamp, and any other metadata stored on the pod.

The block outputs both the `pod_id` and the complete result dictionary. If the pod does not exist, the API error propagates directly to the global error handler.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| pod_id | Pod ID to retrieve | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| pod_id | Unique identifier of the pod | str |
| result | Complete pod object with all metadata | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
- **Tenant Dashboard Display** — Fetch pod details to show a customer's workspace status, creation date, and associated client ID on an admin dashboard.
- **Pre-Action Validation** — Retrieve pod metadata before performing operations like inbox creation to confirm the pod exists and is correctly mapped.
- **Audit Logging** — Pull pod details as part of an automated audit trail that records which tenant workspace was accessed and when.
<!-- END MANUAL -->

---

## Agent Mail List Pod Drafts

### What it is
List all drafts across all inboxes within a pod. View pending emails for a customer.

### How it works
<!-- MANUAL: how_it_works -->
Calls the AgentMail API to retrieve drafts across all inboxes within the specified pod. Optional `limit` and `page_token` parameters control pagination. Only non-empty parameters are sent to the API.

The block returns the list of draft objects, a `count` of drafts in the current page, and a `next_page_token` for fetching subsequent pages. This provides a pod-wide view of unsent emails without needing to query each inbox individually.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| pod_id | Pod ID to list drafts from | str | Yes |
| limit | Maximum number of drafts to return per page (1-100) | int | No |
| page_token | Token from a previous response to fetch the next page | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| drafts | List of draft objects from all inboxes in this pod | List[Dict[str, Any]] |
| count | Number of drafts returned | int |
| next_page_token | Token for the next page. Empty if no more results. | str |

### Possible use case
<!-- MANUAL: use_case -->
- **Draft Review Queue** — Surface all pending drafts across a customer's inboxes so a human reviewer can approve or discard them before sending.
- **Stuck Draft Detection** — Periodically list pod drafts to find emails that have been sitting unsent for too long and alert the responsible agent or operator.
- **Customer Activity Summary** — Include draft counts in a tenant dashboard to show how many outbound emails are queued and awaiting dispatch.
<!-- END MANUAL -->

---

## Agent Mail List Pod Inboxes

### What it is
List all inboxes within a pod. View email accounts scoped to a specific customer.

### How it works
<!-- MANUAL: how_it_works -->
Calls the AgentMail API to list all inboxes belonging to the specified pod. Optional `limit` and `page_token` parameters enable paginated retrieval. Only non-empty parameters are included in the API call.

The block returns the list of inbox objects, a `count` of inboxes in the current page, and a `next_page_token` for fetching additional pages. Each inbox object includes its ID, email address, display name, and other metadata.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| pod_id | Pod ID to list inboxes from | str | Yes |
| limit | Maximum number of inboxes to return per page (1-100) | int | No |
| page_token | Token from a previous response to fetch the next page | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| inboxes | List of inbox objects within this pod | List[Dict[str, Any]] |
| count | Number of inboxes returned | int |
| next_page_token | Token for the next page. Empty if no more results. | str |

### Possible use case
<!-- MANUAL: use_case -->
- **Customer Inbox Inventory** — Display all email addresses belonging to a tenant on their settings page so they can manage or remove unused inboxes.
- **Pre-Deletion Validation** — List a pod's inboxes before attempting to delete the pod, ensuring all inboxes have been removed as required by the API.
- **Multi-Inbox Routing Overview** — Show operators which inboxes exist in a customer's pod so they can configure routing rules for each address.
<!-- END MANUAL -->

---

## Agent Mail List Pod Threads

### What it is
List all conversation threads across all inboxes within a pod. View all email activity for a customer.

### How it works
<!-- MANUAL: how_it_works -->
Calls the AgentMail API to retrieve conversation threads across all inboxes in the specified pod. Supports optional `limit`, `page_token`, and `labels` parameters. When `labels` are provided, only threads matching all specified labels are returned.

The block returns the list of thread objects, a `count` for the current page, and a `next_page_token` for pagination. This gives a unified, cross-inbox view of all email conversations within a customer's workspace.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| pod_id | Pod ID to list threads from | str | Yes |
| limit | Maximum number of threads to return per page (1-100) | int | No |
| page_token | Token from a previous response to fetch the next page | str | No |
| labels | Only return threads matching ALL of these labels | List[str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| threads | List of thread objects from all inboxes in this pod | List[Dict[str, Any]] |
| count | Number of threads returned | int |
| next_page_token | Token for the next page. Empty if no more results. | str |

### Possible use case
<!-- MANUAL: use_case -->
- **Unified Customer Inbox View** — Aggregate all email threads from every inbox in a customer's pod into a single activity feed for support agents or dashboards.
- **Label-Based Ticket Triage** — Filter pod threads by labels like "urgent" or "billing" to route conversations to the appropriate AI agent or human team.
- **Conversation Volume Monitoring** — Periodically list pod threads to track email volume per tenant and trigger alerts when activity spikes or drops.
<!-- END MANUAL -->

---

## Agent Mail List Pods

### What it is
List all tenant pods in your organization. See all customer workspaces at a glance.

### How it works
<!-- MANUAL: how_it_works -->
Calls the AgentMail API to list all pods in your organization. Optional `limit` and `page_token` parameters control pagination. Only non-empty parameters are included in the request.

The block returns a list of pod objects (each containing `pod_id`, `client_id`, creation time, and other metadata), a `count` for the current page, and a `next_page_token` for retrieving additional pages.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| limit | Maximum number of pods to return per page (1-100) | int | No |
| page_token | Token from a previous response to fetch the next page | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| pods | List of pod objects with pod_id, client_id, creation time, etc. | List[Dict[str, Any]] |
| count | Number of pods returned | int |
| next_page_token | Token for the next page. Empty if no more results. | str |

### Possible use case
<!-- MANUAL: use_case -->
- **Admin Tenant Overview** — Display all customer pods on an internal admin dashboard so operators can monitor workspace count and health at a glance.
- **Automated Tenant Reconciliation** — Periodically list all pods and compare against your internal customer database to detect orphaned or missing workspaces.
- **Usage Reporting** — Enumerate all pods to generate per-tenant usage reports or billing summaries based on workspace activity.
<!-- END MANUAL -->

---
