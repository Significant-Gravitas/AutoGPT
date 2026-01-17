# GitHub Triggers
<!-- MANUAL: file_description -->
Blocks for triggering workflows from GitHub webhook events like pull requests, issues, releases, and stars.
<!-- END MANUAL -->

## Github Discussion Trigger

### What it is
This block triggers on GitHub Discussions events. Great for syncing Q&A to Discord or auto-responding to common questions. Note: Discussions must be enabled on the repository.

### How it works
<!-- MANUAL: how_it_works -->
This block creates a webhook subscription to GitHub Discussions events using the GitHub Webhooks API. When a discussion event occurs (created, edited, answered, etc.), GitHub sends a webhook payload that triggers your workflow.

The block parses the webhook payload and extracts discussion details including the title, body, category, state, and the user who triggered the event. Note that GitHub Discussions must be enabled on the repository.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo | Repository to subscribe to.  **Note:** Make sure your GitHub credentials have permissions to create webhooks on this repo. | str | Yes |
| events | The discussion events to subscribe to | Events | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the payload could not be processed | str |
| payload | The complete webhook payload that was received from GitHub. Includes information about the affected resource (e.g. pull request), the event, and the user who triggered the event. | Dict[str, Any] |
| triggered_by_user | Object representing the GitHub user who triggered the event | Dict[str, Any] |
| event | The discussion event that triggered the webhook | str |
| number | The discussion number | int |
| discussion | The full discussion object | Dict[str, Any] |
| discussion_url | URL to the discussion | str |
| title | The discussion title | str |
| body | The discussion body | str |
| category | The discussion category object | Dict[str, Any] |
| category_name | Name of the category | str |
| state | Discussion state | str |

### Possible use case
<!-- MANUAL: use_case -->
**Discord Sync**: Post new discussions to Discord channels to keep the community engaged across platforms.

**Auto-Responder**: Automatically respond to common questions in discussions with helpful resources.

**Q&A Routing**: Route discussion questions to the appropriate team members based on category or content.
<!-- END MANUAL -->

---

## Github Issues Trigger

### What it is
This block triggers on GitHub issues events. Useful for automated triage, notifications, and welcoming first-time contributors.

### How it works
<!-- MANUAL: how_it_works -->
This block creates a webhook subscription to GitHub Issues events. When an issue event occurs (opened, closed, labeled, assigned, etc.), GitHub sends a webhook payload that triggers your workflow.

The block extracts issue details including the title, body, labels, assignees, state, and the user who triggered the event. Use this for automated triage, notifications, and issue management workflows.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo | Repository to subscribe to.  **Note:** Make sure your GitHub credentials have permissions to create webhooks on this repo. | str | Yes |
| events | The issue events to subscribe to | Events | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the payload could not be processed | str |
| payload | The complete webhook payload that was received from GitHub. Includes information about the affected resource (e.g. pull request), the event, and the user who triggered the event. | Dict[str, Any] |
| triggered_by_user | Object representing the GitHub user who triggered the event | Dict[str, Any] |
| event | The issue event that triggered the webhook (e.g., 'opened') | str |
| number | The issue number | int |
| issue | The full issue object | Dict[str, Any] |
| issue_url | URL to the issue | str |
| issue_title | The issue title | str |
| issue_body | The issue body/description | str |
| labels | List of labels on the issue | List[Any] |
| assignees | List of assignees | List[Any] |
| state | Issue state ('open' or 'closed') | str |

### Possible use case
<!-- MANUAL: use_case -->
**Automated Triage**: Automatically label new issues based on keywords in title or description.

**Welcome Messages**: Send welcome messages to first-time contributors when they open their first issue.

**Slack Notifications**: Post notifications to Slack when issues are opened or closed.
<!-- END MANUAL -->

---

## Github Pull Request Trigger

### What it is
This block triggers on pull request events and outputs the event type and payload.

### How it works
<!-- MANUAL: how_it_works -->
This block creates a webhook subscription to GitHub Pull Request events. When a PR event occurs (opened, closed, merged, review requested, etc.), GitHub sends a webhook payload that triggers your workflow.

The block extracts PR details including the number, URL, and full pull request object. This enables automated code review, CI/CD pipelines, and notification workflows.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo | Repository to subscribe to.  **Note:** Make sure your GitHub credentials have permissions to create webhooks on this repo. | str | Yes |
| events | The events to subscribe to | Events | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the payload could not be processed | str |
| payload | The complete webhook payload that was received from GitHub. Includes information about the affected resource (e.g. pull request), the event, and the user who triggered the event. | Dict[str, Any] |
| triggered_by_user | Object representing the GitHub user who triggered the event | Dict[str, Any] |
| event | The PR event that triggered the webhook (e.g. 'opened') | str |
| number | The number of the affected pull request | int |
| pull_request | Object representing the affected pull request | Dict[str, Any] |
| pull_request_url | The URL of the affected pull request | str |

### Possible use case
<!-- MANUAL: use_case -->
**Automated Code Review**: Trigger AI-powered code review when new PRs are opened.

**CI/CD Automation**: Start builds and tests when PRs are created or updated.

**Reviewer Assignment**: Automatically assign reviewers based on files changed or PR author.
<!-- END MANUAL -->

---

## Github Release Trigger

### What it is
This block triggers on GitHub release events. Perfect for automating announcements to Discord, Twitter, or other platforms.

### How it works
<!-- MANUAL: how_it_works -->
This block creates a webhook subscription to GitHub Release events. When a release event occurs (published, created, edited, etc.), GitHub sends a webhook payload that triggers your workflow.

The block extracts release details including tag name, release name, release notes, prerelease flag, and associated assets. Use this to automate announcements and deployment workflows.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo | Repository to subscribe to.  **Note:** Make sure your GitHub credentials have permissions to create webhooks on this repo. | str | Yes |
| events | The release events to subscribe to | Events | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the payload could not be processed | str |
| payload | The complete webhook payload that was received from GitHub. Includes information about the affected resource (e.g. pull request), the event, and the user who triggered the event. | Dict[str, Any] |
| triggered_by_user | Object representing the GitHub user who triggered the event | Dict[str, Any] |
| event | The release event that triggered the webhook (e.g., 'published') | str |
| release | The full release object | Dict[str, Any] |
| release_url | URL to the release page | str |
| tag_name | The release tag name (e.g., 'v1.0.0') | str |
| release_name | Human-readable release name | str |
| body | Release notes/description | str |
| prerelease | Whether this is a prerelease | bool |
| draft | Whether this is a draft release | bool |
| assets | List of release assets/files | List[Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Release Announcements**: Post release announcements to Discord, Twitter, or Slack when new versions are published.

**Changelog Distribution**: Automatically send release notes to mailing lists or documentation sites.

**Deployment Triggers**: Initiate deployment workflows when releases are published.
<!-- END MANUAL -->

---

## Github Star Trigger

### What it is
This block triggers on GitHub star events. Useful for celebrating milestones (e.g., 1k, 10k stars) or tracking engagement.

### How it works
<!-- MANUAL: how_it_works -->
This block creates a webhook subscription to GitHub Star events. When someone stars or unstars your repository, GitHub sends a webhook payload that triggers your workflow.

The block extracts star details including the timestamp, current star count, repository name, and the user who starred. Use this to track engagement and celebrate milestones.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo | Repository to subscribe to.  **Note:** Make sure your GitHub credentials have permissions to create webhooks on this repo. | str | Yes |
| events | The star events to subscribe to | Events | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the payload could not be processed | str |
| payload | The complete webhook payload that was received from GitHub. Includes information about the affected resource (e.g. pull request), the event, and the user who triggered the event. | Dict[str, Any] |
| triggered_by_user | Object representing the GitHub user who triggered the event | Dict[str, Any] |
| event | The star event that triggered the webhook ('created' or 'deleted') | str |
| starred_at | ISO timestamp when the repo was starred (empty if deleted) | str |
| stargazers_count | Current number of stars on the repository | int |
| repository_name | Full name of the repository (owner/repo) | str |
| repository_url | URL to the repository | str |

### Possible use case
<!-- MANUAL: use_case -->
**Milestone Celebrations**: Announce when your repository reaches star milestones (100, 1k, 10k stars).

**Engagement Tracking**: Log star events to track repository popularity over time.

**Thank You Messages**: Send personalized thank you messages to users who star your repository.
<!-- END MANUAL -->

---
