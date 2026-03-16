# Agent Mail Pods
<!-- MANUAL: file_description -->
Blocks for creating, managing, and querying pods in AgentMail. Pods provide multi-tenant isolation between customers — each pod acts as an isolated workspace containing its own inboxes, domains, threads, and drafts. Use pods when building SaaS platforms, agency tools, or AI agent fleets serving multiple customers.
<!-- END MANUAL -->

## Agent Mail Create Pod

### What it is
Create a new pod for multi-tenant customer isolation. Use client_id to map to your internal tenant IDs.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail Create Pod Inbox

### What it is
Create a new email inbox within a pod. The inbox is scoped to the customer workspace.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail Delete Pod

### What it is
Permanently delete a pod. All inboxes and domains must be removed first.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail Get Pod

### What it is
Retrieve details of an existing pod including its client_id mapping and metadata.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail List Pod Drafts

### What it is
List all drafts across all inboxes within a pod. View pending emails for a customer.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail List Pod Inboxes

### What it is
List all inboxes within a pod. View email accounts scoped to a specific customer.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail List Pod Threads

### What it is
List all conversation threads across all inboxes within a pod. View all email activity for a customer.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail List Pods

### What it is
List all tenant pods in your organization. See all customer workspaces at a glance.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
