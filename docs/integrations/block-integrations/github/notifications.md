# GitHub Notifications
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Github Get Notification Thread

### What it is
This block fetches a single GitHub notification thread.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| thread_id | ID of the notification thread | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if fetching the notification thread failed | str |
| notification | The notification thread | NotificationItem |
| title | Title of the notification subject | str |
| reason | Reason you received the notification (e.g. 'mention', 'review_requested', 'subscribed') | str |
| unread | Whether the notification is unread | bool |
| subject_type | Type of the notification subject (e.g. 'Issue', 'PullRequest', 'Release') | str |
| subject_url | URL of the notification subject on GitHub | str |
| repository | Full name of the repository (owner/repo) | str |
| updated_at | ISO 8601 timestamp of the last update to the thread | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github List Notifications

### What it is
This block lists GitHub notifications for the authenticated user, e.g. mentions, review requests, and updates on subscribed threads.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| include_read | Whether to include notifications that are already marked as read | bool | No |
| participating_only | Whether to only include notifications in which you are directly participating or mentioned | bool | No |
| limit | Maximum number of notifications to fetch | int | No |
| repo | Repository to list notifications for. Leave empty to list notifications for all repositories. | str | No |
| since | Only show notifications updated after the given ISO 8601 timestamp | str | No |
| before | Only show notifications updated before the given ISO 8601 timestamp | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if listing notifications failed | str |
| notification | Each notification thread | Notification |
| notifications | List of notification threads | List[NotificationItem] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Mark Notification Thread As Done

### What it is
This block marks a GitHub notification thread as done, removing it from the notification inbox.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| thread_id | ID of the notification thread | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if marking the thread as done failed | str |
| success | Whether the notification thread was marked as done | bool |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Mark Notification Thread As Read

### What it is
This block marks a single GitHub notification thread as read.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| thread_id | ID of the notification thread | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if marking the thread as read failed | str |
| success | Whether the notification thread was marked as read | bool |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Mark Notifications As Read

### What it is
This block marks all GitHub notifications as read, optionally scoped to a single repository.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo | Repository to mark notifications as read for. Leave empty to mark notifications for all repositories. | str | No |
| last_read_at | Only mark notifications updated before the given ISO 8601 timestamp as read. Defaults to the current time. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if marking notifications as read failed | str |
| success | Whether the notifications were marked as read | bool |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Unsubscribe Notification Thread

### What it is
This block unsubscribes you from a GitHub notification thread, muting future notifications unless you are mentioned again.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| thread_id | ID of the notification thread | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if unsubscribing from the thread failed | str |
| success | Whether you were unsubscribed from the notification thread | bool |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
