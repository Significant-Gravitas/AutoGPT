# Github Create Comment Object

### What it is
Creates a comment object for use with GitHub blocks.

### What it does
Creates a comment object for use with GitHub blocks. Note: For review comments, only path, body, and position are used. Side fields are only for standalone PR comments.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
| comment_object | The comment object formatted for GitHub API | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Create PR Review

### What it is
This block creates a review on a GitHub pull request with optional inline comments.

### What it does
This block creates a review on a GitHub pull request with optional inline comments. You can create it as a draft or post immediately. Note: For inline comments, 'position' should be the line number in the diff (starting from the first @@ hunk header).

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo | GitHub repository | str | Yes |
| pr_number | Pull request number | int | Yes |
| body | Body of the review comment | str | Yes |
| event | The review action to perform | "COMMENT" | "APPROVE" | "REQUEST_CHANGES" | No |
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Get PR Review Comments

### What it is
This block gets all review comments from a GitHub pull request or from a specific review.

### What it does
This block gets all review comments from a GitHub pull request or from a specific review.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github List PR Reviews

### What it is
This block lists all reviews for a specified GitHub pull request.

### What it does
This block lists all reviews for a specified GitHub pull request.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Resolve Review Discussion

### What it is
This block resolves or unresolves a review discussion thread on a GitHub pull request.

### What it does
This block resolves or unresolves a review discussion thread on a GitHub pull request.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Submit Pending Review

### What it is
This block submits a pending (draft) review on a GitHub pull request.

### What it does
This block submits a pending (draft) review on a GitHub pull request.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo | GitHub repository | str | Yes |
| pr_number | Pull request number | int | Yes |
| review_id | ID of the pending review to submit | int | Yes |
| event | The review action to perform when submitting | "COMMENT" | "APPROVE" | "REQUEST_CHANGES" | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the review submission failed | str |
| state | State of the submitted review | str |
| html_url | URL of the submitted review | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
