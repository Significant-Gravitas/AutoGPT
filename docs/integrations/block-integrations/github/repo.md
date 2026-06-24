# GitHub Repo
<!-- MANUAL: file_description -->
Blocks for managing GitHub repositories, branches, files, and repository metadata.
<!-- END MANUAL -->

## Github Create Repository

### What it is
This block creates a new GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
This block creates a new GitHub repository under your account using the GitHub API. You can configure visibility (public/private), add a description, and optionally initialize with a README and .gitignore file based on common templates.

The block returns both the web URL for viewing the repository and the clone URL for git operations.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | Name of the repository to create | str | Yes |
| description | Description of the repository | str | No |
| private | Whether the repository should be private | bool | No |
| auto_init | Whether to initialize the repository with a README | bool | No |
| gitignore_template | Git ignore template to use (e.g., Python, Node, Java) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the repository creation failed | str |
| url | URL of the created repository | str |
| clone_url | Git clone URL of the repository | str |

### Possible use case
<!-- MANUAL: use_case -->
**Project Bootstrapping**: Automatically create repositories with standard configuration when starting new projects.

**Template Deployment**: Create pre-configured repositories from templates for team members.

**Automated Workflows**: Generate repositories programmatically as part of onboarding or project management workflows.
<!-- END MANUAL -->

---

## Github Fork Repository

### What it is
This block forks a GitHub repository to your account or an organization.

### How it works
<!-- MANUAL: how_it_works -->
This block forks a GitHub repository by sending a POST request to the GitHub Forks API. You can optionally specify an organization to fork into; if left empty, the fork is created under your personal account.

The block returns the web URL, clone URL, and full name (owner/repo) of the newly created fork.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository to fork | str | Yes |
| organization | Organization to fork into (leave empty to fork to your account) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the fork failed | str |
| url | URL of the forked repository | str |
| clone_url | Git clone URL of the fork | str |
| full_name | Full name of the fork (owner/repo) | str |

### Possible use case
<!-- MANUAL: use_case -->
**Open Source Contributions**: Fork repositories to create your own copy before submitting pull requests with changes.

**Organization Mirroring**: Automatically fork upstream repositories into your organization for internal development.

**Project Scaffolding**: Fork template repositories as starting points for new projects.
<!-- END MANUAL -->

---

## Github Get Repository Info

### What it is
This block retrieves metadata about a GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
This block fetches repository metadata from the GitHub API using the provided repository URL. It returns key information including the repository name, description, default branch, visibility, star/fork/issue counts, and URLs.

The block extracts fields like `stargazers_count`, `forks_count`, and `open_issues_count` from the API response and maps them to the output fields.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if fetching repo info failed | str |
| name | Repository name | str |
| full_name | Full repository name (owner/repo) | str |
| description | Repository description | str |
| default_branch | Default branch name (e.g. main) | str |
| private | Whether the repository is private | bool |
| html_url | Web URL of the repository | str |
| clone_url | Git clone URL | str |
| stars | Number of stars | int |
| forks | Number of forks | int |
| open_issues | Number of open issues | int |

### Possible use case
<!-- MANUAL: use_case -->
**Repository Health Monitoring**: Check star counts, fork counts, and open issues to track project health over time.

**Dependency Assessment**: Retrieve metadata about third-party repositories before adding them as dependencies.

**Automated Reporting**: Collect repository statistics across multiple projects for team dashboards.
<!-- END MANUAL -->

---

## Github List Discussions

### What it is
This block lists recent discussions for a specified GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
This block fetches recent discussions from a GitHub repository using the GitHub GraphQL API. Discussions are a forum-style feature for community conversations separate from issues and PRs.

You can limit the number of discussions retrieved with the num_discussions parameter.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| num_discussions | Number of discussions to fetch | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if listing discussions failed | str |
| discussion | Discussions with their title and URL | Discussion |
| discussions | List of discussions with their title and URL | List[DiscussionItem] |

### Possible use case
<!-- MANUAL: use_case -->
**Community Monitoring**: Track community discussions to identify popular topics or user concerns.

**Q&A Automation**: Monitor discussions for questions that can be answered automatically.

**Content Aggregation**: Collect discussion topics for community newsletters or summaries.
<!-- END MANUAL -->

---

## Github List Releases

### What it is
This block lists all releases for a specified GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves all releases from a GitHub repository. Releases are versioned packages of your software that may include release notes, binaries, and source code archives.

The block returns release information including names and URLs, outputting both individual releases and a complete list.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| release | Releases with their name and file tree browser URL | Release |
| releases | List of releases with their name and file tree browser URL | List[ReleaseItem] |

### Possible use case
<!-- MANUAL: use_case -->
**Version Tracking**: Monitor releases across dependencies to stay current with updates.

**Changelog Compilation**: Gather release information for documentation or announcement purposes.

**Dependency Monitoring**: Track when new versions of libraries your project depends on are released.
<!-- END MANUAL -->

---

## Github List Stargazers

### What it is
This block lists all users who have starred a specified GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves the list of users who have starred a GitHub repository. Stars are a way for users to bookmark or show appreciation for repositories.

Each stargazer entry includes their username and a link to their GitHub profile.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if listing stargazers failed | str |
| stargazer | Stargazers with their username and profile URL | Stargazer |
| stargazers | List of stargazers with their username and profile URL | List[StargazerItem] |

### Possible use case
<!-- MANUAL: use_case -->
**Community Engagement**: Identify and thank users who have starred your repository.

**Growth Analytics**: Track repository popularity over time by monitoring star growth.

**User Research**: Analyze who is interested in your project based on their profiles.
<!-- END MANUAL -->

---

## Github List Tags

### What it is
This block lists all tags for a specified GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves all git tags from a GitHub repository. Tags are typically used to mark release points or significant milestones in the repository history.

Each tag includes its name and a URL to browse the repository files at that tag.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| tag | Tags with their name and file tree browser URL | Tag |
| tags | List of tags with their name and file tree browser URL | List[TagItem] |

### Possible use case
<!-- MANUAL: use_case -->
**Version Enumeration**: List all versions of a project to check for available updates.

**Release Verification**: Confirm that tags exist for expected release versions.

**Historical Code Access**: Find tags to access the codebase at specific historical points.
<!-- END MANUAL -->

---

## Github Star Repository

### What it is
This block stars a GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
This block stars a GitHub repository by sending a PUT request to the GitHub Starring API (`/user/starred/{owner}/{repo}`). Starring is a way to bookmark repositories and show appreciation for projects.

The block returns a success status message upon completion.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository to star | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if starring failed | str |
| status | Status of the star operation | str |

### Possible use case
<!-- MANUAL: use_case -->
**Bookmarking Repositories**: Automatically star repositories that match certain criteria for later reference.

**Community Engagement**: Star repositories from contributors as part of an automated thank-you workflow.

**Interest Tracking**: Programmatically star repositories in specific topics to build a curated collection.
<!-- END MANUAL -->

---
