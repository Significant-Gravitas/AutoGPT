# Pull Requests

## GitHub List Pull Requests

### What it is
A block that retrieves a list of pull requests from a specified GitHub repository.

### What it does
This block fetches all open pull requests for a given GitHub repository and provides their titles and URLs.

### How it works
It connects to the GitHub API using the provided credentials and repository URL, then retrieves the list of pull requests and formats the information for easy viewing.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | GitHub authentication details to access the repository |
| Repository URL | The URL of the GitHub repository to fetch pull requests from |

### Outputs
| Output | Description |
|--------|-------------|
| Pull Request | A list of pull requests, each containing: |
| - Title | The title of the pull request |
| - URL | The web address of the pull request |
| Error | An error message if the operation fails |

### Possible use case
A development team leader wants to quickly review all open pull requests in their project repository to prioritize code reviews.

---

## GitHub Make Pull Request

### What it is
A block that creates a new pull request in a specified GitHub repository.

### What it does
This block allows users to create a new pull request by providing details such as title, body, and branch information.

### How it works
It uses the GitHub API to create a new pull request with the given information, including the source and target branches for the changes.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | GitHub authentication details to access the repository |
| Repository URL | The URL of the GitHub repository where the pull request will be created |
| Title | The title of the new pull request |
| Body | The description or content of the pull request |
| Head | The name of the branch containing the changes |
| Base | The name of the branch you want to merge the changes into |

### Outputs
| Output | Description |
|--------|-------------|
| Number | The unique identifier of the created pull request |
| URL | The web address of the newly created pull request |
| Error | An error message if the pull request creation fails |

### Possible use case
A developer has finished working on a new feature in a separate branch and wants to create a pull request to merge their changes into the main branch for review.

---

## GitHub Read Pull Request

### What it is
A block that retrieves detailed information about a specific GitHub pull request.

### What it does
This block fetches and provides comprehensive information about a given pull request, including its title, body, author, and optionally, the changes made.

### How it works
It connects to the GitHub API using the provided credentials and pull request URL, then retrieves and formats the requested information.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | GitHub authentication details to access the repository |
| Pull Request URL | The URL of the specific GitHub pull request to read |
| Include PR Changes | An option to include the actual changes made in the pull request |

### Outputs
| Output | Description |
|--------|-------------|
| Title | The title of the pull request |
| Body | The description or content of the pull request |
| Author | The username of the person who created the pull request |
| Changes | A list of changes made in the pull request (if requested) |
| Error | An error message if reading the pull request fails |

### Possible use case
A code reviewer wants to get a comprehensive overview of a pull request, including its description and changes, before starting the review process.

---

## GitHub Assign PR Reviewer

### What it is
A block that assigns a reviewer to a specific GitHub pull request.

### What it does
This block allows users to assign a designated reviewer to a given pull request in a GitHub repository.

### How it works
It uses the GitHub API to add the specified user as a reviewer for the given pull request.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | GitHub authentication details to access the repository |
| Pull Request URL | The URL of the specific GitHub pull request to assign a reviewer to |
| Reviewer | The username of the GitHub user to be assigned as a reviewer |

### Outputs
| Output | Description |
|--------|-------------|
| Status | A message indicating whether the reviewer was successfully assigned |
| Error | An error message if the reviewer assignment fails |

### Possible use case
A project manager wants to assign a specific team member to review a newly created pull request for a critical feature.

---

## GitHub Unassign PR Reviewer

### What it is
A block that removes an assigned reviewer from a specific GitHub pull request.

### What it does
This block allows users to unassign a previously designated reviewer from a given pull request in a GitHub repository.

### How it works
It uses the GitHub API to remove the specified user from the list of reviewers for the given pull request.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | GitHub authentication details to access the repository |
| Pull Request URL | The URL of the specific GitHub pull request to unassign a reviewer from |
| Reviewer | The username of the GitHub user to be unassigned as a reviewer |

### Outputs
| Output | Description |
|--------|-------------|
| Status | A message indicating whether the reviewer was successfully unassigned |
| Error | An error message if the reviewer unassignment fails |

### Possible use case
A team lead realizes that an assigned reviewer is unavailable and wants to remove them from a pull request to reassign it to another team member.

---

## GitHub List PR Reviewers

### What it is
A block that retrieves a list of all assigned reviewers for a specific GitHub pull request.

### What it does
This block fetches and provides information about all the reviewers currently assigned to a given pull request in a GitHub repository.

### How it works
It connects to the GitHub API using the provided credentials and pull request URL, then retrieves and formats the list of assigned reviewers.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | GitHub authentication details to access the repository |
| Pull Request URL | The URL of the specific GitHub pull request to list reviewers for |

### Outputs
| Output | Description |
|--------|-------------|
| Reviewer | A list of assigned reviewers, each containing: |
| - Username | The GitHub username of the reviewer |
| - URL | The profile URL of the reviewer |
| Error | An error message if listing the reviewers fails |

### Possible use case
A project coordinator wants to check who is currently assigned to review a specific pull request to ensure all necessary team members are involved in the code review process.