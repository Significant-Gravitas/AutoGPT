# GitHub Reviews
<!-- MANUAL: file_description -->
Blocks for creating and managing GitHub pull request reviews and review comments.
<!-- END MANUAL -->

## Github Create Comment Object

### What it is
Creates a comment object for use with GitHub blocks. Note: For review comments, only path, body, and position are used. Side fields are only for standalone PR comments.

### How it works
<!-- MANUAL: how_it_works -->
This block creates a structured comment object that can be used with GitHub review blocks. It formats the comment data according to GitHub API requirements, including file path, body text, and position information.

For review comments, only path, body, and position fields are used. The side, start_line, and start_side fields are only applicable for standalone PR comments, not review comments.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| path | The file path to comment on | str | Yes |
| body | The comment text | str | Yes |
| position | Position in the diff (line number from first @@ hunk). Use this OR line. | int | No |
| line | Line number in the file (will be used as position if position not provided) | int | No |
| side | Side of the diff to comment on (NOTE: Only for standalone comments, not review comments) | str | No |
| start_line | Start line for multi-line comments (NOTE: Only for standalone comments, not review comments) | int | No |
| start_side | Side for the start of multi-line comments (NOTE: Only for standalone comments, not review comments) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| comment_object | The comment object formatted for GitHub API | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Automated Code Review**: Generate comment objects for automated review systems that analyze code changes.

**Batch Comments**: Create multiple comment objects to submit together in a single review.

**Template Responses**: Build reusable comment templates for common code review feedback patterns.
<!-- END MANUAL -->

---

## Github Create PR Review

### What it is
This block creates a review on a GitHub pull request with optional inline comments. You can create it as a draft or post immediately. Note: For inline comments, 'position' should be the line number in the diff (starting from the first @@ hunk header).

### How it works
<!-- MANUAL: how_it_works -->
This block creates a code review on a GitHub pull request using the Reviews API. Reviews can include a summary comment and optionally inline comments on specific lines of code in the diff.

You can create reviews as drafts (pending) for later submission, or post them immediately with an action: COMMENT for neutral feedback, APPROVE to approve the changes, or REQUEST_CHANGES to block merging until addressed.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo | GitHub repository | str | Yes |
| pr_number | Pull request number | int | Yes |
| body | Body of the review comment | str | Yes |
| event | The review action to perform | "COMMENT" \| "APPROVE" \| "REQUEST_CHANGES" | No |
| create_as_draft | Create the review as a draft (pending) or post it immediately | bool | No |
| comments | Optional inline comments to add to specific files/lines. Note: Only path, body, and position are supported. Position is line number in diff from first @@ hunk. | List[ReviewComment] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the review creation failed | str |
| review_id | ID of the created review | int |
| state | State of the review (e.g., PENDING, COMMENTED, APPROVED, CHANGES_REQUESTED) | str |
| html_url | URL of the created review | str |

### Possible use case
<!-- MANUAL: use_case -->
**Automated Code Review**: Submit AI-generated code reviews with inline comments on specific lines.

**Review Workflows**: Create structured reviews as part of automated CI/CD pipelines.

**Approval Automation**: Automatically approve PRs that pass all automated checks and criteria.
<!-- END MANUAL -->

---

## Github Get PR Review Comments

### What it is
This block gets all review comments from a GitHub pull request or from a specific review.

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves review comments from a GitHub pull request. Review comments are inline comments made on specific lines of code during code review, distinct from general issue-style comments.

You can get all review comments on the PR, or filter to comments from a specific review by providing the review ID. Each comment includes metadata like the file path, line number, author, and comment body.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo | GitHub repository | str | Yes |
| pr_number | Pull request number | int | Yes |
| review_id | ID of a specific review to get comments from (optional) | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| comment | Individual review comment with details | Comment |
| comments | List of all review comments on the pull request | List[CommentItem] |

### Possible use case
<!-- MANUAL: use_case -->
**Review Analysis**: Extract all review comments to analyze feedback patterns or generate summaries.

**Comment Tracking**: Monitor which review feedback has been addressed by comparing comments to code changes.

**Documentation**: Collect review discussions for documentation or knowledge base purposes.
<!-- END MANUAL -->

---

## Github List PR Reviews

### What it is
This block lists all reviews for a specified GitHub pull request.

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves all reviews submitted on a GitHub pull request. It returns information about each review including the reviewer, their verdict (approve, request changes, or comment), and the review body.

Use this to check approval status, see who has reviewed, or analyze the review history of a pull request.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo | GitHub repository | str | Yes |
| pr_number | Pull request number | int | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| review | Individual review with details | Review |
| reviews | List of all reviews on the pull request | List[ReviewItem] |

### Possible use case
<!-- MANUAL: use_case -->
**Approval Status Check**: Verify that required reviewers have approved before proceeding with merge.

**Review Metrics**: Track review participation and response times across team members.

**Merge Readiness**: Check review states to determine if a PR meets merge requirements.
<!-- END MANUAL -->

---

## Github Resolve Review Discussion

### What it is
This block resolves or unresolves a review discussion thread on a GitHub pull request.

### How it works
<!-- MANUAL: how_it_works -->
This block resolves or unresolves a review discussion thread on a GitHub pull request using the GraphQL API. Resolved discussions are collapsed in the GitHub UI, indicating the feedback has been addressed.

Specify the comment ID of the thread to resolve. Set resolve to true to mark as resolved, or false to reopen the discussion.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo | GitHub repository | str | Yes |
| pr_number | Pull request number | int | Yes |
| comment_id | ID of the review comment to resolve/unresolve | int | Yes |
| resolve | Whether to resolve (true) or unresolve (false) the discussion | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the operation was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Automated Resolution**: Mark discussions as resolved when automated systems verify the feedback was addressed.

**Review Cleanup**: Bulk resolve outdated discussions that no longer apply after significant refactoring.

**Review Management**: Programmatically manage discussion states as part of review workflows.
<!-- END MANUAL -->

---

## Github Submit Pending Review

### What it is
This block submits a pending (draft) review on a GitHub pull request.

### How it works
<!-- MANUAL: how_it_works -->
This block submits a pending (draft) review on a GitHub pull request. Draft reviews allow you to compose multiple inline comments before publishing them together as a cohesive review.

When submitting, choose the review event: COMMENT for general feedback, APPROVE to approve the PR, or REQUEST_CHANGES to request modifications before merging.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo | GitHub repository | str | Yes |
| pr_number | Pull request number | int | Yes |
| review_id | ID of the pending review to submit | int | Yes |
| event | The review action to perform when submitting | "COMMENT" \| "APPROVE" \| "REQUEST_CHANGES" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the review submission failed | str |
| state | State of the submitted review | str |
| html_url | URL of the submitted review | str |

### Possible use case
<!-- MANUAL: use_case -->
**Batch Review Submission**: Build up multiple comments in a draft, then submit them all at once.

**Review Finalization**: Complete the review process after adding all inline comments and deciding on the verdict.

**Two-Phase Review**: Create draft reviews for internal review before officially submitting to the PR author.
<!-- END MANUAL -->

---
