# GitHub Issues

## Github Comment

### What it is
A block that posts comments on GitHub issues or pull requests.

### What it does
This block allows users to add comments to existing GitHub issues or pull requests using the GitHub API.

### How it works
The block takes the GitHub credentials, the URL of the issue or pull request, and the comment text as inputs. It then sends a request to the GitHub API to post the comment on the specified issue or pull request.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | GitHub authentication information |
| Issue URL | The URL of the GitHub issue or pull request where the comment will be posted |
| Comment | The text content of the comment to be posted |

### Outputs
| Output | Description |
|--------|-------------|
| ID | The unique identifier of the created comment |
| URL | The direct link to the posted comment on GitHub |
| Error | Any error message if the comment posting fails |

### Possible use case
Automating responses to issues in a GitHub repository, such as thanking contributors for their submissions or providing status updates on reported bugs.

---

## Github Make Issue

### What it is
A block that creates new issues on GitHub repositories.

### What it does
This block allows users to create new issues in a specified GitHub repository with a title and body content.

### How it works
The block takes the GitHub credentials, repository URL, issue title, and issue body as inputs. It then sends a request to the GitHub API to create a new issue with the provided information.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | GitHub authentication information |
| Repo URL | The URL of the GitHub repository where the issue will be created |
| Title | The title of the new issue |
| Body | The main content or description of the new issue |

### Outputs
| Output | Description |
|--------|-------------|
| Number | The issue number assigned by GitHub |
| URL | The direct link to the newly created issue on GitHub |
| Error | Any error message if the issue creation fails |

### Possible use case
Automatically creating issues for bug reports or feature requests submitted through an external system or form.

---

## Github Read Issue

### What it is
A block that retrieves information about a specific GitHub issue.

### What it does
This block fetches the details of a given GitHub issue, including its title, body content, and the user who created it.

### How it works
The block takes the GitHub credentials and the issue URL as inputs. It then sends a request to the GitHub API to fetch the issue's details and returns the relevant information.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | GitHub authentication information |
| Issue URL | The URL of the GitHub issue to be read |

### Outputs
| Output | Description |
|--------|-------------|
| Title | The title of the issue |
| Body | The main content or description of the issue |
| User | The username of the person who created the issue |
| Error | Any error message if reading the issue fails |

### Possible use case
Gathering information about reported issues for analysis or to display on a dashboard.

---

## Github List Issues

### What it is
A block that retrieves a list of issues from a GitHub repository.

### What it does
This block fetches all open issues from a specified GitHub repository and provides their titles and URLs.

### How it works
The block takes the GitHub credentials and repository URL as inputs. It then sends a request to the GitHub API to fetch the list of issues and returns their details.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | GitHub authentication information |
| Repo URL | The URL of the GitHub repository to list issues from |

### Outputs
| Output | Description |
|--------|-------------|
| Issue | A list of issues, each containing: |
| - Title | The title of the issue |
| - URL | The direct link to the issue on GitHub |
| Error | Any error message if listing the issues fails |

### Possible use case
Creating a summary of open issues for a project status report or displaying them on a project management dashboard.

---

## Github Add Label

### What it is
A block that adds a label to a GitHub issue or pull request.

### What it does
This block allows users to add a specified label to an existing GitHub issue or pull request.

### How it works
The block takes the GitHub credentials, the URL of the issue or pull request, and the label to be added as inputs. It then sends a request to the GitHub API to add the label to the specified issue or pull request.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | GitHub authentication information |
| Issue URL | The URL of the GitHub issue or pull request to add the label to |
| Label | The name of the label to be added |

### Outputs
| Output | Description |
|--------|-------------|
| Status | A message indicating whether the label was successfully added |
| Error | Any error message if adding the label fails |

### Possible use case
Automatically categorizing issues based on their content or assigning priority labels to newly created issues.

---

## Github Remove Label

### What it is
A block that removes a label from a GitHub issue or pull request.

### What it does
This block allows users to remove a specified label from an existing GitHub issue or pull request.

### How it works
The block takes the GitHub credentials, the URL of the issue or pull request, and the label to be removed as inputs. It then sends a request to the GitHub API to remove the label from the specified issue or pull request.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | GitHub authentication information |
| Issue URL | The URL of the GitHub issue or pull request to remove the label from |
| Label | The name of the label to be removed |

### Outputs
| Output | Description |
|--------|-------------|
| Status | A message indicating whether the label was successfully removed |
| Error | Any error message if removing the label fails |

### Possible use case
Updating the status of issues as they progress through a workflow, such as removing a "In Progress" label when an issue is completed.

---

## Github Assign Issue

### What it is
A block that assigns a user to a GitHub issue.

### What it does
This block allows users to assign a specific GitHub user to an existing issue.

### How it works
The block takes the GitHub credentials, the URL of the issue, and the username of the person to be assigned as inputs. It then sends a request to the GitHub API to assign the specified user to the issue.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | GitHub authentication information |
| Issue URL | The URL of the GitHub issue to assign |
| Assignee | The username of the person to be assigned to the issue |

### Outputs
| Output | Description |
|--------|-------------|
| Status | A message indicating whether the issue was successfully assigned |
| Error | Any error message if assigning the issue fails |

### Possible use case
Automatically assigning new issues to team members based on their expertise or workload.

---

## Github Unassign Issue

### What it is
A block that unassigns a user from a GitHub issue.

### What it does
This block allows users to remove a specific GitHub user's assignment from an existing issue.

### How it works
The block takes the GitHub credentials, the URL of the issue, and the username of the person to be unassigned as inputs. It then sends a request to the GitHub API to remove the specified user's assignment from the issue.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | GitHub authentication information |
| Issue URL | The URL of the GitHub issue to unassign |
| Assignee | The username of the person to be unassigned from the issue |

### Outputs
| Output | Description |
|--------|-------------|
| Status | A message indicating whether the issue was successfully unassigned |
| Error | Any error message if unassigning the issue fails |

### Possible use case
Automatically unassigning issues that have been inactive for a certain period or when reassigning workload among team members.