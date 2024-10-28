# Repository

## GitHub List Tags

### What it is
A block that retrieves and lists all tags for a specified GitHub repository.

### What it does
This block fetches all tags associated with a given GitHub repository and provides their names and URLs.

### How it works
The block connects to the GitHub API using provided credentials, sends a request to retrieve tag information for the specified repository, and then processes the response to extract tag names and URLs.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | GitHub authentication credentials required to access the repository |
| Repository URL | The URL of the GitHub repository to fetch tags from |

### Outputs
| Output | Description |
|--------|-------------|
| Tag | Information about each tag, including its name and URL |
| Error | Any error message if the tag listing process fails |

### Possible use case
A developer wants to quickly view all release tags for a project to identify the latest version or track the project's release history.

---

## GitHub List Branches

### What it is
A block that retrieves and lists all branches for a specified GitHub repository.

### What it does
This block fetches all branches associated with a given GitHub repository and provides their names and URLs.

### How it works
The block authenticates with the GitHub API, sends a request to get branch information for the specified repository, and then processes the response to extract branch names and URLs.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | GitHub authentication credentials required to access the repository |
| Repository URL | The URL of the GitHub repository to fetch branches from |

### Outputs
| Output | Description |
|--------|-------------|
| Branch | Information about each branch, including its name and URL |
| Error | Any error message if the branch listing process fails |

### Possible use case
A project manager wants to review all active branches in a repository to track ongoing development efforts and feature implementations.

---

## GitHub List Discussions

### What it is
A block that retrieves and lists recent discussions for a specified GitHub repository.

### What it does
This block fetches a specified number of recent discussions from a given GitHub repository and provides their titles and URLs.

### How it works
The block uses the GitHub GraphQL API to request discussion data for the specified repository, processes the response, and extracts discussion titles and URLs.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | GitHub authentication credentials required to access the repository |
| Repository URL | The URL of the GitHub repository to fetch discussions from |
| Number of Discussions | The number of recent discussions to retrieve (default is 5) |

### Outputs
| Output | Description |
|--------|-------------|
| Discussion | Information about each discussion, including its title and URL |
| Error | Any error message if the discussion listing process fails |

### Possible use case
A community manager wants to monitor recent discussions in a project's repository to identify trending topics or issues that need attention.

---

## GitHub List Releases

### What it is
A block that retrieves and lists all releases for a specified GitHub repository.

### What it does
This block fetches all releases associated with a given GitHub repository and provides their names and URLs.

### How it works
The block connects to the GitHub API, sends a request to get release information for the specified repository, and then processes the response to extract release names and URLs.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | GitHub authentication credentials required to access the repository |
| Repository URL | The URL of the GitHub repository to fetch releases from |

### Outputs
| Output | Description |
|--------|-------------|
| Release | Information about each release, including its name and URL |
| Error | Any error message if the release listing process fails |

### Possible use case
A user wants to view all official releases of a software project to choose the appropriate version for installation or to track the project's release history.

---

## GitHub Read File

### What it is
A block that reads the content of a specified file from a GitHub repository.

### What it does
This block retrieves the content of a specified file from a given GitHub repository, providing both the raw and decoded text content along with the file size.

### How it works
The block authenticates with the GitHub API, sends a request to fetch the specified file's content, and then processes the response to provide the file's raw content, decoded text content, and size.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | GitHub authentication credentials required to access the repository |
| Repository URL | The URL of the GitHub repository containing the file |
| File Path | The path to the file within the repository |
| Branch | The branch name to read from (defaults to "master") |

### Outputs
| Output | Description |
|--------|-------------|
| Text Content | The content of the file decoded as UTF-8 text |
| Raw Content | The raw base64-encoded content of the file |
| Size | The size of the file in bytes |
| Error | Any error message if the file reading process fails |

### Possible use case
A developer wants to quickly view the contents of a configuration file or source code file in a GitHub repository without having to clone the entire repository.

---

## GitHub Read Folder

### What it is
A block that reads the content of a specified folder from a GitHub repository.

### What it does
This block retrieves the list of files and directories within a specified folder from a given GitHub repository.

### How it works
The block connects to the GitHub API, sends a request to fetch the contents of the specified folder, and then processes the response to provide information about files and directories within that folder.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | GitHub authentication credentials required to access the repository |
| Repository URL | The URL of the GitHub repository containing the folder |
| Folder Path | The path to the folder within the repository |
| Branch | The branch name to read from (defaults to "master") |

### Outputs
| Output | Description |
|--------|-------------|
| File | Information about each file in the folder, including its name, path, and size |
| Directory | Information about each directory in the folder, including its name and path |
| Error | Any error message if the folder reading process fails |

### Possible use case
A project manager wants to explore the structure of a repository or specific folder to understand the organization of files and directories without cloning the entire repository.

---

## GitHub Make Branch

### What it is
A block that creates a new branch in a GitHub repository.

### What it does
This block creates a new branch in a specified GitHub repository, based on an existing source branch.

### How it works
The block authenticates with the GitHub API, retrieves the latest commit SHA of the source branch, and then creates a new branch pointing to that commit.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | GitHub authentication credentials required to access the repository |
| Repository URL | The URL of the GitHub repository where the new branch will be created |
| New Branch | The name of the new branch to create |
| Source Branch | The name of the existing branch to use as the starting point for the new branch |

### Outputs
| Output | Description |
|--------|-------------|
| Status | A message indicating the success of the branch creation operation |
| Error | Any error message if the branch creation process fails |

### Possible use case
A developer wants to start working on a new feature and needs to create a new branch based on the current state of the main development branch.

---

## GitHub Delete Branch

### What it is
A block that deletes a specified branch from a GitHub repository.

### What it does
This block removes a specified branch from a given GitHub repository.

### How it works
The block authenticates with the GitHub API and sends a delete request for the specified branch.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | GitHub authentication credentials required to access the repository |
| Repository URL | The URL of the GitHub repository containing the branch to be deleted |
| Branch | The name of the branch to delete |

### Outputs
| Output | Description |
|--------|-------------|
| Status | A message indicating the success of the branch deletion operation |
| Error | Any error message if the branch deletion process fails |

### Possible use case
After merging a feature branch into the main development branch, a developer wants to clean up the repository by removing the now-obsolete feature branch.