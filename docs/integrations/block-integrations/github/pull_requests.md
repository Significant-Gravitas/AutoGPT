# GitHub Pull Requests
<!-- MANUAL: file_description -->
Blocks for managing GitHub pull requests including creating, reading, listing PRs, and assigning or unassigning reviewers.
<!-- END MANUAL -->

## Github Assign PR Reviewer

### What it is
This block assigns a reviewer to a specified GitHub pull request.

### How it works
<!-- MANUAL: how_it_works -->
This block requests a code review from a specific user on a GitHub pull request. It uses the GitHub API to add the specified username to the list of requested reviewers, triggering a notification to that user.

The reviewer must have access to the repository. Organization members can typically be assigned as reviewers on any repository they have at least read access to.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| pr_url | URL of the GitHub pull request | str | Yes |
| reviewer | Username of the reviewer to assign | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the reviewer assignment failed | str |
| status | Status of the reviewer assignment operation | str |

### Possible use case
<!-- MANUAL: use_case -->
**Automated Code Review Assignment**: Automatically assign reviewers based on the files changed or the PR author.

**Round-Robin Reviews**: Distribute code review load evenly across team members.

**Expertise-Based Routing**: Assign reviewers who are experts in the specific area of code being modified.
<!-- END MANUAL -->

---

## Github List PR Reviewers

### What it is
This block lists all reviewers for a specified GitHub pull request.

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves the list of requested reviewers for a GitHub pull request. It queries the GitHub API to fetch all users who have been requested to review the PR, returning their usernames and profile URLs.

This includes both pending review requests and users who have already submitted reviews.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| pr_url | URL of the GitHub pull request | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if listing reviewers failed | str |
| reviewer | Reviewers with their username and profile URL | Reviewer |
| reviewers | List of reviewers with their username and profile URL | List[ReviewerItem] |

### Possible use case
<!-- MANUAL: use_case -->
**Review Status Monitoring**: Check which reviewers have been assigned to a PR and send reminders to those who haven't responded.

**Workflow Validation**: Verify that required reviewers have been assigned before a PR can be merged.

**Team Dashboard**: Display reviewer assignments across multiple PRs for team visibility.
<!-- END MANUAL -->

---

## Github List Pull Requests

### What it is
This block lists all pull requests for a specified GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
This block fetches all open pull requests from a GitHub repository. It queries the GitHub API and returns a list of PRs with their titles and URLs, outputting both individual PRs and a complete list.

The block returns open pull requests by default, allowing you to monitor pending code changes in a repository.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if listing pull requests failed | str |
| pull_request | PRs with their title and URL | Pull Request |
| pull_requests | List of pull requests with their title and URL | List[PRItem] |

### Possible use case
<!-- MANUAL: use_case -->
**PR Dashboard**: Create a dashboard showing all open pull requests across your repositories.

**Merge Queue Monitoring**: Track pending PRs to prioritize code reviews and identify bottlenecks.

**Stale PR Detection**: List PRs to identify those that have been open too long and need attention.
<!-- END MANUAL -->

---

## Github Make Pull Request

### What it is
This block creates a new pull request on a specified GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
This block creates a new pull request on a GitHub repository. It uses the GitHub API to submit a PR from your source branch (head) to the target branch (base), with the specified title and description.

For cross-repository PRs, format the head branch as "username:branch". The branches must exist and have divergent commits for the PR to be created successfully.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| title | Title of the pull request | str | Yes |
| body | Body of the pull request | str | Yes |
| head | The name of the branch where your changes are implemented. For cross-repository pull requests in the same network, namespace head with a user like this: username:branch. | str | Yes |
| base | The name of the branch you want the changes pulled into. | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the pull request creation failed | str |
| number | Number of the created pull request | int |
| url | URL of the created pull request | str |

### Possible use case
<!-- MANUAL: use_case -->
**Automated Releases**: Create PRs automatically when a release branch is ready to merge to main.

**Dependency Updates**: Programmatically create PRs for dependency updates after testing passes.

**Feature Flags**: Automatically create PRs to enable feature flags in configuration files.
<!-- END MANUAL -->

---

## Github Read Pull Request

### What it is
This block reads the body, title, user, and changes of a specified GitHub pull request.

### How it works
<!-- MANUAL: how_it_works -->
This block reads the details of a GitHub pull request including its title, description, author, and optionally the code diff. It fetches this information via the GitHub API using your credentials.

When include_pr_changes is enabled, the block also retrieves the full diff of all changes in the PR, which can be useful for code review automation or analysis.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| pr_url | URL of the GitHub pull request | str | Yes |
| include_pr_changes | Whether to include the changes made in the pull request | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if reading the pull request failed | str |
| title | Title of the pull request | str |
| body | Body of the pull request | str |
| author | User who created the pull request | str |
| changes | Changes made in the pull request | str |

### Possible use case
<!-- MANUAL: use_case -->
**Automated Code Review**: Read PR content and changes to perform automated code analysis or send to AI for review.

**Changelog Generation**: Extract PR titles and descriptions to automatically compile release notes.

**PR Summarization**: Read PR details to generate summaries for stakeholder updates.
<!-- END MANUAL -->

---

## Github Unassign PR Reviewer

### What it is
This block unassigns a reviewer from a specified GitHub pull request.

### How it works
<!-- MANUAL: how_it_works -->
This block removes a reviewer from a GitHub pull request's review request list. It uses the GitHub API to remove the specified user from pending reviewers, which stops further review notifications to that user.

This is useful for reassigning reviews or removing reviewers who are unavailable.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| pr_url | URL of the GitHub pull request | str | Yes |
| reviewer | Username of the reviewer to unassign | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the reviewer unassignment failed | str |
| status | Status of the reviewer unassignment operation | str |

### Possible use case
<!-- MANUAL: use_case -->
**Reviewer Reassignment**: Remove unavailable reviewers and replace them with available team members.

**Load Balancing**: Unassign reviewers who have too many pending reviews.

**Vacation Coverage**: Automatically remove reviewers who are out of office.
<!-- END MANUAL -->

---
