# GitHub Repo
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Github Create Repository

### What it is
This block creates a new GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Fork Repository

### What it is
This block forks a GitHub repository to your account or an organization.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Get Repository Info

### What it is
This block retrieves metadata about a GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github List Discussions

### What it is
This block lists recent discussions for a specified GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github List Releases

### What it is
This block lists all releases for a specified GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github List Stargazers

### What it is
This block lists all users who have starred a specified GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github List Tags

### What it is
This block lists all tags for a specified GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Star Repository

### What it is
This block stars a GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
