# AgentMail Pods
<!-- MANUAL: file_description -->
Blocks for creating, managing, and querying pods in AgentMail. Pods provide multi-tenant isolation between customers — each pod acts as an isolated workspace containing its own inboxes, domains, threads, and drafts. Use pods when building SaaS platforms, agency tools, or AI agent fleets serving multiple customers.
<!-- END MANUAL -->

## Create Pod

### What it is
A block that creates a new pod for multi-tenant customer isolation.

### How it works
<!-- MANUAL: how_it_works -->
The block creates an isolated workspace for one customer or tenant. Use `client_id` to map pods to your internal tenant IDs for idempotent creation — safe to retry without creating duplicates. This lets you access the pod by your own ID instead of AgentMail's pod_id.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| client_id | Your internal tenant/customer ID for idempotent mapping | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| pod_id | Unique identifier of the created pod | str |
| result | Complete pod object with all metadata | dict |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**SaaS Multi-Tenancy**: Create a pod for each customer to isolate their email data from other customers.

**Agency Platform**: Provision isolated email workspaces for each client managed by your agency.
<!-- END MANUAL -->

---

## Get Pod

### What it is
A block that retrieves details of an existing pod by its ID.

### How it works
<!-- MANUAL: how_it_works -->
The block fetches the pod metadata including its client_id mapping and creation timestamp. Use this to verify a pod exists or retrieve its configuration.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| pod_id | Pod ID to retrieve | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| pod_id | Unique identifier of the pod | str |
| result | Complete pod object with all metadata | dict |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Pod Verification**: Check that a customer's pod exists and is correctly configured before performing operations.

**Tenant Mapping**: Retrieve the client_id associated with a pod to map it back to your internal customer records.
<!-- END MANUAL -->

---

## List Pods

### What it is
A block that lists all pods in your AgentMail organization.

### How it works
<!-- MANUAL: how_it_works -->
The block retrieves a paginated list of all tenant pods with their metadata. Use this to see all customer workspaces at a glance.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| limit | Maximum number of pods to return per page (1-100) | int | No |
| page_token | Token from a previous response to fetch the next page | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| pods | List of pod objects with pod_id, client_id, creation time, etc. | List[dict] |
| count | Number of pods returned | int |
| next_page_token | Token for the next page. Empty if no more results. | str |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Admin Overview**: Display all customer pods in an admin dashboard for monitoring and management.

**Billing Reconciliation**: List all pods to reconcile tenant usage with billing records.
<!-- END MANUAL -->

---

## Delete Pod

### What it is
A block that permanently deletes a pod after all its inboxes and domains have been removed.

### How it works
<!-- MANUAL: how_it_works -->
The block deletes the pod permanently. You cannot delete a pod that still contains inboxes or domains — delete all child resources first, then delete the pod.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| pod_id | Pod ID to permanently delete (must have no inboxes or domains) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| success | True if the pod was successfully deleted | bool |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Customer Offboarding**: Delete a customer's pod after they've been offboarded and all their inboxes removed.

**Environment Cleanup**: Remove test or staging pods that are no longer needed.
<!-- END MANUAL -->

---

## List Pod Inboxes

### What it is
A block that lists all inboxes within a specific pod (customer workspace).

### How it works
<!-- MANUAL: how_it_works -->
The block returns only the inboxes belonging to this pod, providing tenant-scoped visibility. Supports pagination for pods with many inboxes.
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
| inboxes | List of inbox objects within this pod | List[dict] |
| count | Number of inboxes returned | int |
| next_page_token | Token for the next page. Empty if no more results. | str |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Customer Dashboard**: Show a customer all their email inboxes within their isolated workspace.

**Resource Management**: List inboxes in a pod to check capacity before creating new ones.
<!-- END MANUAL -->

---

## List Pod Threads

### What it is
A block that lists all conversation threads across all inboxes within a pod.

### How it works
<!-- MANUAL: how_it_works -->
The block returns threads from every inbox in the pod. Use for building per-customer dashboards showing all email activity, or for supervisor agents monitoring a customer's conversations. Supports label filtering and pagination.
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
| threads | List of thread objects from all inboxes in this pod | List[dict] |
| count | Number of threads returned | int |
| next_page_token | Token for the next page. Empty if no more results. | str |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Customer Activity View**: Show all email conversations for a specific customer across all their inboxes.

**Tenant Monitoring**: A supervisor agent monitors all conversations in a customer's pod to detect issues or escalations.
<!-- END MANUAL -->

---

## List Pod Drafts

### What it is
A block that lists all drafts across all inboxes within a pod.

### How it works
<!-- MANUAL: how_it_works -->
The block returns pending drafts from every inbox in the pod. Use for per-customer approval dashboards or monitoring scheduled sends within a customer workspace.
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
| drafts | List of draft objects from all inboxes in this pod | List[dict] |
| count | Number of drafts returned | int |
| next_page_token | Token for the next page. Empty if no more results. | str |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Customer Draft Review**: Show all pending drafts for a customer so they can review and approve before sending.

**Scheduled Send Dashboard**: Monitor all scheduled emails within a customer's workspace.
<!-- END MANUAL -->

---

## Create Pod Inbox

### What it is
A block that creates a new email inbox within a specific pod (customer workspace).

### How it works
<!-- MANUAL: how_it_works -->
The block creates an inbox that is automatically scoped to the pod and inherits its isolation guarantees. If username and domain are not provided, AgentMail auto-generates a unique address. The inbox can only be accessed within the context of its pod.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| pod_id | Pod ID to create the inbox in | str | Yes |
| username | Local part of the email address (e.g. 'support'). Leave empty to auto-generate. | str | No |
| domain | Email domain (e.g. 'mydomain.com'). Defaults to agentmail.to if empty. | str | No |
| display_name | Friendly name shown in the 'From' field | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| inbox_id | Unique identifier of the created inbox | str |
| email_address | Full email address of the inbox | str |
| result | Complete inbox object with all metadata | dict |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Customer Email Provisioning**: Create a dedicated support inbox for a customer within their isolated pod.

**Multi-Agent Pods**: Provision multiple agent inboxes within a customer's pod, each handling a different function (sales, support, billing).
<!-- END MANUAL -->
