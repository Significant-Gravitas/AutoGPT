# Github Create File

### What it is
This block creates a new file in a GitHub repository.

### What it does
This block creates a new file in a GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Create Repository

### What it is
This block creates a new GitHub repository.

### What it does
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

## Github Delete Branch

### What it is
This block deletes a specified branch.

### What it does
This block deletes a specified branch.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github List Branches

### What it is
This block lists all branches for a specified GitHub repository.

### What it does
This block lists all branches for a specified GitHub repository.

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
| branch | Branches with their name and file tree browser URL | Branch |
| branches | List of branches with their name and file tree browser URL | List[BranchItem] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github List Discussions

### What it is
This block lists recent discussions for a specified GitHub repository.

### What it does
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

### What it does
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

### What it does
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

### What it does
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

## Github Make Branch

### What it is
This block creates a new branch from a specified source branch.

### What it does
This block creates a new branch from a specified source branch.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Read File

### What it is
This block reads the content of a specified file from a GitHub repository.

### What it does
This block reads the content of a specified file from a GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Read Folder

### What it is
This block reads the content of a specified folder from a GitHub repository.

### What it does
This block reads the content of a specified folder from a GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Update File

### What it is
This block updates an existing file in a GitHub repository.

### What it does
This block updates an existing file in a GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
