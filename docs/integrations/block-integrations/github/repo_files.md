# GitHub Repo Files
<!-- MANUAL: file_description -->
Blocks for reading, creating, updating, and searching files and directories in GitHub repositories.
<!-- END MANUAL -->

## Github Create File

### What it is
This block creates a new file in a GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
This block creates a new file in a GitHub repository using the Contents API. It base64-encodes the provided content and sends a PUT request with the file path, content, branch, and commit message.

The block returns the URL of the created file and the SHA of the resulting commit.
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
**Configuration Bootstrapping**: Create configuration files like `.env` or `config.yaml` in new repositories automatically.

**Documentation Generation**: Write generated documentation files directly to a repository.

**Template Deployment**: Add boilerplate files such as CI configs or linting rules to repositories.
<!-- END MANUAL -->

---

## Github Get Repository Tree

### What it is
This block lists the entire file tree of a GitHub repository recursively.

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves the file tree of a GitHub repository using the Git Trees API. When the recursive option is enabled, it returns every file and directory in the entire repository; otherwise it returns only the top-level entries.

Each entry includes its path, type (blob for files, tree for directories), size, and SHA. The block also indicates whether the tree was truncated due to exceeding GitHub's size limits.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| branch | Branch name to get the tree from | str | No |
| recursive | Whether to recursively list the entire tree | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if getting tree failed | str |
| entry | A file or directory in the tree | Tree Entry |
| entries | List of all files and directories in the tree | List[TreeEntry] |
| truncated | Whether the tree was truncated due to size | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Repository Structure Analysis**: Map out the full directory structure of a repository for documentation or visualization.

**File Discovery**: Find specific files by browsing the tree before reading their contents.

**Size Auditing**: Identify large files in a repository by inspecting file sizes in the tree.
<!-- END MANUAL -->

---

## Github Read File

### What it is
This block reads the content of a specified file from a GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
This block reads a file from a GitHub repository using the Contents API. It fetches the file at the specified path and branch, returning both the raw base64-encoded content and the UTF-8 decoded text content along with the file size.

If multiple entries exist at the path (e.g., a file and a symlink), the block selects the file entry. It raises an error if the path points to a directory instead of a file.
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
| error | Error message if reading the file failed | str |
| text_content | Content of the file (decoded as UTF-8 text) | str |
| raw_content | Raw base64-encoded content of the file | str |
| size | The size of the file (in bytes) | int |

### Possible use case
<!-- MANUAL: use_case -->
**Configuration Reading**: Read configuration files like `package.json` or `pyproject.toml` from repositories for analysis.

**Code Review Automation**: Fetch specific source files to run automated checks or AI-powered code review.

**Content Extraction**: Read README or documentation files from repositories to aggregate information.
<!-- END MANUAL -->

---

## Github Read Folder

### What it is
This block reads the content of a specified folder from a GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
This block lists the contents of a folder in a GitHub repository using the Contents API. It fetches all entries at the specified path and separates them into files (with name, path, and size) and directories (with name and path).

The block raises an error if the path does not point to a directory. Only the immediate children of the folder are returned, not nested contents.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| folder_path | Path to the folder in the repository | str | Yes |
| branch | Branch name to read from (defaults to main) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if reading the folder failed | str |
| file | Files in the folder | FileEntry |
| dir | Directories in the folder | DirEntry |

### Possible use case
<!-- MANUAL: use_case -->
**Directory Exploration**: Browse folder contents to understand a repository's structure without cloning it.

**File Listing**: List files in a specific directory to find relevant source files or configurations.

**Project Inventory**: Enumerate files and subdirectories in a project folder for automated processing.
<!-- END MANUAL -->

---

## Github Search Code

### What it is
This block searches for code in GitHub repositories.

### How it works
<!-- MANUAL: how_it_works -->
This block searches for code across GitHub using the Code Search API. It supports GitHub's code search syntax and can optionally be scoped to a specific repository using the `repo` parameter (in `owner/repo` format).

Each result includes the file name, path, repository, URL, and relevance score. The block also returns the total count of matching results.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Search query (GitHub code search syntax) | str | Yes |
| repo | Restrict search to a repository (owner/repo format, optional) | str | No |
| per_page | Number of results to return (max 100) | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if search failed | str |
| result | A code search result | Result |
| results | List of code search results | List[SearchResult] |
| total_count | Total number of matching results | int |

### Possible use case
<!-- MANUAL: use_case -->
**Code Pattern Detection**: Search for specific function calls, imports, or patterns across repositories.

**Vulnerability Scanning**: Find usage of deprecated APIs or known vulnerable code patterns.

**Cross-Repository Analysis**: Locate how a library or API is used across multiple projects.
<!-- END MANUAL -->

---

## Github Update File

### What it is
This block updates an existing file in a GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
This block updates an existing file using the GitHub Contents API. It first fetches the current file to obtain its SHA (required by the API to prevent conflicts), then sends a PUT request with the new base64-encoded content, commit message, and branch.

The block returns the URL of the updated file and the SHA of the new commit.
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
**Configuration Updates**: Modify configuration files like version numbers or feature flags in repositories.

**Automated Maintenance**: Update dependency files or lock files as part of automated workflows.

**Content Management**: Edit documentation or content files stored in GitHub repositories.
<!-- END MANUAL -->

---
