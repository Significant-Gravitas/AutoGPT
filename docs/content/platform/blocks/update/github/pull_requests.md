

## GitHub Pull Request Trigger

### What it is
A specialized block that monitors GitHub repositories and triggers actions based on pull request events.

### What it does
This block acts as a listener for various pull request-related events in a GitHub repository. It captures the event details and provides structured information about pull request activities, making it easier to automate workflows based on GitHub pull request actions.

### How it works
The block establishes a webhook connection with a specified GitHub repository. When pull request events occur that match the configured event filters, the block captures the event data and outputs relevant information about the pull request and the user who triggered the event.

### Inputs
- Credentials: GitHub authentication credentials required to access the repository
- Repository: The GitHub repository to monitor (in format "owner/repo")
- Events Filter: A set of toggles for different pull request events:
  - Opened: When a new pull request is created
  - Edited: When a pull request is modified
  - Closed: When a pull request is closed
  - Reopened: When a closed pull request is reopened
  - Synchronize: When the pull request's branch is updated
  - Assigned/Unassigned: When reviewers are assigned or removed
  - Labeled/Unlabeled: When labels are added or removed
  - Draft Status Changes: When converting to/from draft status
  - Lock Status: When the pull request is locked or unlocked
  - Review Status: When review is requested or removed
  - Auto-merge: When auto-merge is enabled or disabled
  - Milestone Changes: When milestones are added or removed

### Outputs
- Payload: Complete webhook data received from GitHub
- Triggered By User: Information about the GitHub user who initiated the event
- Event: The specific type of pull request event that occurred
- Number: The pull request's identifying number
- Pull Request: Detailed information about the affected pull request
- Pull Request URL: Direct link to the pull request on GitHub
- Error: Any error message if the webhook payload processing fails

### Possible use cases
- Automating code review workflows by triggering actions when new pull requests are opened
- Notifying team members when pull requests are ready for review
- Tracking pull request statistics and activities
- Creating custom notifications for specific pull request events
- Integrating pull request activities with project management tools
- Automating deployment processes based on pull request status changes

