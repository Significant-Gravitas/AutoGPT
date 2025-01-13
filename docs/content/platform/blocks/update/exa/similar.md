
<file_name>autogpt_platform/backend/backend/blocks/github/issues.md</file_name>

# GitHub Issues Management Blocks Documentation

## GitHub Comment
### What it is
A block that allows users to post comments on GitHub issues or pull requests.

### What it does
Posts a new comment on a specified GitHub issue or pull request and returns information about the created comment.

### How it works
The block takes your comment and GitHub credentials, posts it to the specified issue or pull request, and provides you with the comment's ID and URL.

### Inputs
- Credentials: Your GitHub authentication credentials
- Issue URL: The URL of the GitHub issue or pull request where you want to comment
- Comment: The text content of your comment

### Outputs
- ID: The unique identifier of your posted comment
- URL: Direct link to your comment on GitHub
- Error: Any error message if the comment posting fails

### Possible use case
Automating responses to user-reported issues or providing status updates on pull requests.

## GitHub Make Issue
### What it is
A block that creates new issues in GitHub repositories.

### What it does
Creates a new issue with a specified title and body content in a GitHub repository.

### How it works
Takes your issue details and credentials, creates a new issue in the specified repository, and returns the issue number and URL.

### Inputs
- Credentials: Your GitHub authentication credentials
- Repository URL: The URL of the GitHub repository where you want to create the issue
- Title: The title of your new issue
- Body: The main content of your issue

### Outputs
- Number: The issue number assigned by GitHub
- URL: Direct link to the created issue
- Error: Any error message if the issue creation fails

### Possible use case
Automatically creating bug reports or feature requests based on user feedback.

## GitHub Read Issue
### What it is
A block that retrieves and reads information from existing GitHub issues.

### What it does
Fetches and provides detailed information about a specific GitHub issue.

### How it works
Uses the provided issue URL to retrieve the issue's details and returns its title, body content, and creator information.

### Inputs
- Credentials: Your GitHub authentication credentials
- Issue URL: The URL of the GitHub issue you want to read

### Outputs
- Title: The issue's title
- Body: The main content of the issue
- User: The username of the person who created the issue
- Error: Any error message if reading the issue fails

### Possible use case
Monitoring specific issues or gathering information for status reports.

## GitHub List Issues
### What it is
A block that retrieves a list of all issues from a GitHub repository.

### What it does
Fetches and provides a list of all issues in a specified repository.

### How it works
Retrieves all issues from the specified repository and returns their titles and URLs.

### Inputs
- Credentials: Your GitHub authentication credentials
- Repository URL: The URL of the GitHub repository whose issues you want to list

### Outputs
- Issue: Contains title and URL for each issue
- Error: Any error message if listing issues fails

### Possible use case
Creating issue summaries or monitoring overall repository activity.

## GitHub Add Label
### What it is
A block that adds labels to GitHub issues or pull requests.

### What it does
Applies a specified label to an existing GitHub issue or pull request.

### How it works
Takes a label name and applies it to the specified issue or pull request.

### Inputs
- Credentials: Your GitHub authentication credentials
- Issue URL: The URL of the issue or pull request you want to label
- Label: The name of the label to add

### Outputs
- Status: Confirmation message about the label addition
- Error: Any error message if adding the label fails

### Possible use case
Automatically categorizing issues based on their content or priority.

## GitHub Remove Label
### What it is
A block that removes labels from GitHub issues or pull requests.

### What it does
Removes a specified label from an existing GitHub issue or pull request.

### How it works
Takes a label name and removes it from the specified issue or pull request.

### Inputs
- Credentials: Your GitHub authentication credentials
- Issue URL: The URL of the issue or pull request you want to remove the label from
- Label: The name of the label to remove

### Outputs
- Status: Confirmation message about the label removal
- Error: Any error message if removing the label fails

### Possible use case
Updating issue categories when their status changes.

## GitHub Assign Issue
### What it is
A block that assigns users to GitHub issues.

### What it does
Assigns a specified user to an existing GitHub issue.

### How it works
Takes a username and assigns that user to the specified issue.

### Inputs
- Credentials: Your GitHub authentication credentials
- Issue URL: The URL of the issue you want to assign
- Assignee: The username of the person to assign

### Outputs
- Status: Confirmation message about the assignment
- Error: Any error message if the assignment fails

### Possible use case
Automatically assigning issues to team members based on their expertise.

## GitHub Unassign Issue
### What it is
A block that removes user assignments from GitHub issues.

### What it does
Removes a specified user's assignment from an existing GitHub issue.

### How it works
Takes a username and removes their assignment from the specified issue.

### Inputs
- Credentials: Your GitHub authentication credentials
- Issue URL: The URL of the issue you want to unassign
- Assignee: The username of the person to unassign

### Outputs
- Status: Confirmation message about the unassignment
- Error: Any error message if the unassignment fails

### Possible use case
Automatically reassigning issues when team members are unavailable or when tasks need to be redistributed.
