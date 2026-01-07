# Github Discussion Trigger

### What it is
This block triggers on GitHub Discussions events.

### What it does
This block triggers on GitHub Discussions events. Great for syncing Q&A to Discord or auto-responding to common questions. Note: Discussions must be enabled on the repository.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo | Repository to subscribe to.

**Note:** Make sure your GitHub credentials have permissions to create webhooks on this repo. | str | Yes |
| events | The discussion events to subscribe to | Events | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the payload could not be processed | str |
| payload | The complete webhook payload that was received from GitHub. Includes information about the affected resource (e.g. pull request), the event, and the user who triggered the event. | Dict[str, True] |
| triggered_by_user | Object representing the GitHub user who triggered the event | Dict[str, True] |
| event | The discussion event that triggered the webhook | str |
| number | The discussion number | int |
| discussion | The full discussion object | Dict[str, True] |
| discussion_url | URL to the discussion | str |
| title | The discussion title | str |
| body | The discussion body | str |
| category | The discussion category object | Dict[str, True] |
| category_name | Name of the category | str |
| state | Discussion state | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Issues Trigger

### What it is
This block triggers on GitHub issues events.

### What it does
This block triggers on GitHub issues events. Useful for automated triage, notifications, and welcoming first-time contributors.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo | Repository to subscribe to.

**Note:** Make sure your GitHub credentials have permissions to create webhooks on this repo. | str | Yes |
| events | The issue events to subscribe to | Events | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the payload could not be processed | str |
| payload | The complete webhook payload that was received from GitHub. Includes information about the affected resource (e.g. pull request), the event, and the user who triggered the event. | Dict[str, True] |
| triggered_by_user | Object representing the GitHub user who triggered the event | Dict[str, True] |
| event | The issue event that triggered the webhook (e.g., 'opened') | str |
| number | The issue number | int |
| issue | The full issue object | Dict[str, True] |
| issue_url | URL to the issue | str |
| issue_title | The issue title | str |
| issue_body | The issue body/description | str |
| labels | List of labels on the issue | List[Any] |
| assignees | List of assignees | List[Any] |
| state | Issue state ('open' or 'closed') | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Pull Request Trigger

### What it is
This block triggers on pull request events and outputs the event type and payload.

### What it does
This block triggers on pull request events and outputs the event type and payload.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo | Repository to subscribe to.

**Note:** Make sure your GitHub credentials have permissions to create webhooks on this repo. | str | Yes |
| events | The events to subscribe to | Events | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the payload could not be processed | str |
| payload | The complete webhook payload that was received from GitHub. Includes information about the affected resource (e.g. pull request), the event, and the user who triggered the event. | Dict[str, True] |
| triggered_by_user | Object representing the GitHub user who triggered the event | Dict[str, True] |
| event | The PR event that triggered the webhook (e.g. 'opened') | str |
| number | The number of the affected pull request | int |
| pull_request | Object representing the affected pull request | Dict[str, True] |
| pull_request_url | The URL of the affected pull request | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Release Trigger

### What it is
This block triggers on GitHub release events.

### What it does
This block triggers on GitHub release events. Perfect for automating announcements to Discord, Twitter, or other platforms.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo | Repository to subscribe to.

**Note:** Make sure your GitHub credentials have permissions to create webhooks on this repo. | str | Yes |
| events | The release events to subscribe to | Events | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the payload could not be processed | str |
| payload | The complete webhook payload that was received from GitHub. Includes information about the affected resource (e.g. pull request), the event, and the user who triggered the event. | Dict[str, True] |
| triggered_by_user | Object representing the GitHub user who triggered the event | Dict[str, True] |
| event | The release event that triggered the webhook (e.g., 'published') | str |
| release | The full release object | Dict[str, True] |
| release_url | URL to the release page | str |
| tag_name | The release tag name (e.g., 'v1.0.0') | str |
| release_name | Human-readable release name | str |
| body | Release notes/description | str |
| prerelease | Whether this is a prerelease | bool |
| draft | Whether this is a draft release | bool |
| assets | List of release assets/files | List[Any] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Star Trigger

### What it is
This block triggers on GitHub star events.

### What it does
This block triggers on GitHub star events. Useful for celebrating milestones (e.g., 1k, 10k stars) or tracking engagement.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo | Repository to subscribe to.

**Note:** Make sure your GitHub credentials have permissions to create webhooks on this repo. | str | Yes |
| events | The star events to subscribe to | Events | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the payload could not be processed | str |
| payload | The complete webhook payload that was received from GitHub. Includes information about the affected resource (e.g. pull request), the event, and the user who triggered the event. | Dict[str, True] |
| triggered_by_user | Object representing the GitHub user who triggered the event | Dict[str, True] |
| event | The star event that triggered the webhook ('created' or 'deleted') | str |
| starred_at | ISO timestamp when the repo was starred (empty if deleted) | str |
| stargazers_count | Current number of stars on the repository | int |
| repository_name | Full name of the repository (owner/repo) | str |
| repository_url | URL to the repository | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
