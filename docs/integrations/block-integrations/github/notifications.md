# GitHub Notifications
<!-- MANUAL: file_description -->
Blocks for reading and clearing the GitHub notification inbox — the same feed as the bell icon on github.com — so an agent can triage what it is being asked to look at and then clear those threads behind itself.
<!-- END MANUAL -->

## Github Get Notification Thread

### What it is
This block fetches a single GitHub notification thread.

### How it works
<!-- MANUAL: how_it_works -->
Fetches one thread from `/notifications/threads/{thread_id}` and normalises it into the same `NotificationItem` shape the list block emits, so both can feed the same downstream nodes. The subject URL GitHub returns is an API URL, and the block rewrites it per path segment into its github.com equivalent: `/pulls/{n}` becomes `/pull/{n}`, `/commits/{sha}` becomes `/commit/{sha}`, and a release — which the API addresses by numeric id but the web addresses by tag — degrades to the repository's releases index. Anything that is not a repo-scoped API URL is passed through untouched. Note that a thread id is the notification's own id, not the issue or pull request number.
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
A triage agent walks an inbox listing and re-fetches each thread before acting on it. A notification may have been read, resolved or closed elsewhere since the listing was taken, and acting on stale state means replying to something a colleague already handled.
<!-- END MANUAL -->

---

## Github List Notifications

### What it is
This block lists GitHub notifications for the authenticated user, e.g. mentions, review requests, and updates on subscribed threads.

### How it works
<!-- MANUAL: how_it_works -->
Reads the notification inbox, either globally or scoped to a single repository, and follows pagination until `limit` items are collected rather than returning only the first page. The two endpoints differ in page size — the global `/notifications` caps a page at 50 while the repo-scoped one allows the usual 100 — and the block uses whichever applies. `repo` accepts either `{owner}/{repo}` or a full repository URL. By default only unread threads come back; `include_read` sends `all=true`, and `participating_only` narrows to threads where you are directly involved rather than merely watching. `since` and `before` bound the results by last-update time. Each item is flattened to a `NotificationItem` whose `subject_url` is a browsable github.com link, not the API URL.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| include_read | Whether to include notifications that are already marked as read | bool | No |
| participating_only | Whether to only include notifications in which you are directly participating or mentioned | bool | No |
| limit | Maximum number of notifications to fetch | int | No |
| repo | Repository to list notifications for, as '{owner}/{repo}' or a full repository URL. Leave empty to list notifications for all repositories. | str | No |
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
A morning triage agent lists notifications with `participating_only` enabled, groups them by `reason` — `review_requested`, `mention`, `assign` — and posts one ranked digest to Slack, so a maintainer sees what actually needs them instead of scrolling the bell menu.
<!-- END MANUAL -->

---

## Github Mark Notification Thread As Done

### What it is
This block marks a GitHub notification thread as done, removing it from the notification inbox.

### How it works
<!-- MANUAL: how_it_works -->
Sends `DELETE` to `/notifications/threads/{thread_id}`. Despite the verb, nothing is deleted: GitHub's "done" state removes the thread from your inbox and marks it read, while the underlying issue, pull request or discussion is untouched, and new activity on that subject can surface it again. This is distinct from unsubscribing, which stops future notifications for the subject entirely.
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
Once a triage agent has summarised and filed a batch of notifications, it marks each one done so the next run starts from a clean inbox instead of re-reporting the same threads.
<!-- END MANUAL -->

---

## Github Mark Notification Thread As Read

### What it is
This block marks a single GitHub notification thread as read.

### How it works
<!-- MANUAL: how_it_works -->
Sends `PATCH` to `/notifications/threads/{thread_id}`. The thread keeps its place in the inbox but loses its unread flag, so a later listing without `include_read` skips it. Use this rather than marking the thread done when it should stay visible.
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
An agent that auto-answers a known subset of mentions marks only those threads read, leaving everything it could not handle unread and still in the inbox for a human.
<!-- END MANUAL -->

---

## Github Mark Notifications As Read

### What it is
This block marks all GitHub notifications as read, optionally scoped to a single repository.

### How it works
<!-- MANUAL: how_it_works -->
Sends `PUT` to the notifications endpoint, either globally or for one repository; `repo` accepts `{owner}/{repo}` or a full repository URL, and leaving it empty clears the whole inbox. `last_read_at` restricts the sweep to threads updated before that timestamp and defaults to now, which is what makes this safe to run against a busy inbox: anything arriving while the request is in flight keeps its unread state rather than being silently cleared.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo | Repository to mark notifications as read for, as '{owner}/{repo}' or a full repository URL. Leave empty to mark notifications for all repositories. | str | No |
| last_read_at | Only mark notifications updated before the given ISO 8601 timestamp as read. Defaults to the current time. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if marking notifications as read failed | str |
| success | Whether the notifications were marked as read | bool |

### Possible use case
<!-- MANUAL: use_case -->
A weekly hygiene agent passes the start of the week as `last_read_at`, clearing a backlog of stale notifications in one call while leaving the last few days' worth untouched.
<!-- END MANUAL -->

---

## Github Unsubscribe Notification Thread

### What it is
This block unsubscribes you from a GitHub notification thread, muting future notifications unless you are mentioned again.

### How it works
<!-- MANUAL: how_it_works -->
Sends `DELETE` to `/notifications/threads/{thread_id}/subscription`, a different endpoint from the one that marks a thread done. This ends your subscription to the subject, so further activity on that issue or pull request produces no new notifications at all. The effect persists until you subscribe again or someone mentions you directly.
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
A long-running, high-traffic issue that an agent was added to but has no stake in gets unsubscribed after a single triage pass, so its every comment stops re-entering the inbox.
<!-- END MANUAL -->

---
