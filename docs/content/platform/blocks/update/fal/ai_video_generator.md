

# GitHub Pull Request Blocks Documentation

## GitHub List Pull Requests

### What it is
A tool that retrieves a list of all pull requests from a specified GitHub repository.

### What it does
Fetches and displays all pull requests from a given GitHub repository, showing their titles and URLs.

### How it works
Connects to GitHub using provided credentials, retrieves pull request data from the specified repository, and presents it in an organized format.

### Inputs
- Credentials: GitHub authentication credentials with repository access permissions
- Repository URL: The URL of the GitHub repository to fetch pull requests from

### Outputs
- Pull Request: Information about each pull request, including:
  - Title: The name/title of the pull request
  - URL: Direct link to the pull request
- Error: Any error message if the operation fails

### Possible use case
A development team manager wanting to review all active pull requests in a project repository.

## GitHub Make Pull Request

### What it is
A tool for creating new pull requests in a GitHub repository.

### What it does
Creates a new pull request with specified details such as title, description, and branch information.

### How it works
Uses GitHub credentials to create a new pull request with the provided information, linking specified branches for code review.

### Inputs
- Credentials: GitHub authentication credentials with repository access
- Repository URL: The URL of the target GitHub repository
- Title: The title for the new pull request
- Body: Detailed description of the changes
- Head: The branch containing the changes
- Base: The branch you want to merge changes into

### Outputs
- Number: The assigned pull request number
- URL: Direct link to the created pull request
- Error: Any error message if creation fails

### Possible use case
A developer completing a feature and wanting to submit their changes for review.

## GitHub Read Pull Request

### What it is
A tool that retrieves detailed information about a specific pull request.

### What it does
Fetches and displays comprehensive information about a pull request, including its content and changes.

### How it works
Retrieves pull request metadata and optionally includes the actual code changes made in the pull request.

### Inputs
- Credentials: GitHub authentication credentials
- Pull Request URL: Direct link to the pull request
- Include PR Changes: Option to include code changes (true/false)

### Outputs
- Title: The pull request's title
- Body: The full description/content
- Author: Username of the person who created it
- Changes: Detailed code changes (if requested)
- Error: Any error message if reading fails

### Possible use case
A code reviewer wanting to examine the details of a specific pull request before review.

## GitHub Assign PR Reviewer

### What it is
A tool for assigning reviewers to a pull request.

### What it does
Adds a specified user as a reviewer to a GitHub pull request.

### How it works
Uses GitHub's review system to officially assign a user as a reviewer for the pull request.

### Inputs
- Credentials: GitHub authentication credentials
- Pull Request URL: Link to the target pull request
- Reviewer: Username of the person to assign as reviewer

### Outputs
- Status: Confirmation of successful assignment
- Error: Any error message if assignment fails

### Possible use case
A team lead assigning team members to review specific pull requests.

## GitHub Unassign PR Reviewer

### What it is
A tool for removing reviewers from a pull request.

### What it does
Removes a specified user from the reviewer list of a pull request.

### How it works
Removes the assigned reviewer from the pull request's review list using GitHub's API.

### Inputs
- Credentials: GitHub authentication credentials
- Pull Request URL: Link to the target pull request
- Reviewer: Username of the reviewer to remove

### Outputs
- Status: Confirmation of successful removal
- Error: Any error message if unassignment fails

### Possible use case
Reassigning reviews when a team member becomes unavailable.

## GitHub List PR Reviewers

### What it is
A tool that shows all currently assigned reviewers for a pull request.

### What it does
Retrieves and displays a list of all users assigned to review a specific pull request.

### How it works
Fetches reviewer information from GitHub and presents it in an organized format.

### Inputs
- Credentials: GitHub authentication credentials
- Pull Request URL: Link to the target pull request

### Outputs
- Reviewer: Information about each reviewer, including:
  - Username: The reviewer's GitHub username
  - URL: Link to the reviewer's GitHub profile
- Error: Any error message if listing fails

### Possible use case
Checking who is currently assigned to review a specific pull request.

