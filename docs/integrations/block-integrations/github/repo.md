# GitHub Repo
<!-- MANUAL: file_description -->
Blocks for managing GitHub repositories, branches, files, and repository metadata.
<!-- END MANUAL -->

## Github Create File

### What it is
This block creates a new file in a GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
This block creates a new file in a GitHub repository using the GitHub Contents API. It commits the file with the specified content to the chosen branch (or the default branch if not specified).

The commit message can be customized, and the block returns the URL of the created file along with the commit SHA for tracking purposes.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| file_path | Path where the file should be created | str | Yes |
| content | Content to write to the file | str | Yes |
| branch | Branch where the file should be created | str | No |
| commit_message | Message for the commit | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the file creation failed | str |
| url | URL of the created file | str |
| sha | SHA of the commit | str |

### Possible use case
<!-- MANUAL: use_case -->
**Configuration Deployment**: Automatically add configuration files to repositories during project setup.

**Documentation Generation**: Create markdown files or documentation pages programmatically.

**Template Deployment**: Add boilerplate files like LICENSE, .gitignore, or CI configs to repositories.
<!-- END MANUAL -->

---

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

## Github Delete Branch

### What it is
This block deletes a specified branch.

### How it works
<!-- MANUAL: how_it_works -->
This block deletes a specified branch from a GitHub repository using the GitHub References API. The branch is permanently removed, so use with caution—this cannot be undone without re-pushing the branch.

Protected branches cannot be deleted unless protection rules are first removed.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| branch | Name of the branch to delete | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the branch deletion failed | str |
| status | Status of the branch deletion operation | str |

### Possible use case
<!-- MANUAL: use_case -->
**Post-Merge Cleanup**: Automatically delete feature branches after they've been merged.

**Stale Branch Management**: Clean up old or abandoned branches to keep the repository tidy.

**CI/CD Automation**: Delete temporary branches created during build or deployment processes.
<!-- END MANUAL -->

---

## Github List Branches

### What it is
This block lists all branches for a specified GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves all branches from a GitHub repository. It queries the GitHub API and returns each branch with its name and a URL to browse the files at that branch.

This provides visibility into all development streams in a repository.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| branch | Branches with their name and file tree browser URL | Branch |
| branches | List of branches with their name and file tree browser URL | List[BranchItem] |

### Possible use case
<!-- MANUAL: use_case -->
**Branch Inventory**: Create a dashboard showing all active branches across repositories.

**Naming Convention Validation**: Check branch names against team conventions.

**Active Development Tracking**: Monitor which branches exist to track parallel development efforts.
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

## Github Make Branch

### What it is
This block creates a new branch from a specified source branch.

### How it works
<!-- MANUAL: how_it_works -->
This block creates a new branch in a GitHub repository based on an existing source branch. It uses the GitHub References API to create a new ref pointing to the same commit as the source branch.

The new branch immediately contains all the code from the source branch at the time of creation.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| new_branch | Name of the new branch | str | Yes |
| source_branch | Name of the source branch | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the branch creation failed | str |
| status | Status of the branch creation operation | str |

### Possible use case
<!-- MANUAL: use_case -->
**Feature Branch Creation**: Automatically create feature branches from main when work begins.

**Release Preparation**: Create release branches from development when ready to stabilize.

**Hotfix Workflows**: Quickly create hotfix branches from production for urgent fixes.
<!-- END MANUAL -->

---

## Github Read File

### What it is
This block reads the content of a specified file from a GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
This block reads the contents of a file from a GitHub repository using the Contents API. You can specify which branch to read from, defaulting to the repository's default branch.

The block returns both the decoded text content (for text files) and the raw base64-encoded content, along with the file size.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| file_path | Path to the file in the repository | str | Yes |
| branch | Branch to read from | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| text_content | Content of the file (decoded as UTF-8 text) | str |
| raw_content | Raw base64-encoded content of the file | str |
| size | The size of the file (in bytes) | int |

### Possible use case
<!-- MANUAL: use_case -->
**Configuration Reading**: Fetch configuration files from repositories for processing or validation.

**Code Analysis**: Read source files for automated analysis, linting, or documentation generation.

**Version Comparison**: Compare file contents across different branches or versions.
<!-- END MANUAL -->

---

## Github Read Folder

### What it is
This block reads the content of a specified folder from a GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
This block lists the contents of a folder in a GitHub repository. It returns separate outputs for files and directories found in the specified path, allowing you to explore the repository structure.

You can specify which branch to read from; it defaults to master if not specified.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| folder_path | Path to the folder in the repository | str | Yes |
| branch | Branch name to read from (defaults to master) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if reading the folder failed | str |
| file | Files in the folder | FileEntry |
| dir | Directories in the folder | DirEntry |

### Possible use case
<!-- MANUAL: use_case -->
**Repository Exploration**: Browse repository structure to understand project organization.

**File Discovery**: Find specific file types in directories for batch processing.

**Directory Monitoring**: Check for expected files in specific locations.
<!-- END MANUAL -->

---

## Github Update File

### What it is
This block updates an existing file in a GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
This block updates an existing file in a GitHub repository using the Contents API. It creates a new commit with the updated file content. The block automatically handles the required SHA of the existing file.

You can customize the commit message and specify which branch to update.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| file_path | Path to the file to update | str | Yes |
| content | New content for the file | str | Yes |
| branch | Branch containing the file | str | No |
| commit_message | Message for the commit | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| url | URL of the updated file | str |
| sha | SHA of the commit | str |

### Possible use case
<!-- MANUAL: use_case -->
**Configuration Updates**: Programmatically update configuration files in repositories.

**Version Bumping**: Automatically update version numbers in package files.

**Documentation Sync**: Update documentation files based on code changes.
<!-- END MANUAL -->

---
