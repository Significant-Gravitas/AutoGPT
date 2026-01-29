# Linear Comment
<!-- MANUAL: file_description -->
Blocks for creating and managing comments on Linear issues.
<!-- END MANUAL -->

## Linear Create Comment

### What it is
Creates a new comment on a Linear issue

### How it works
<!-- MANUAL: how_it_works -->
This block creates a new comment on a Linear issue using the Linear GraphQL API. Provide the issue ID and comment text, and the block posts the comment and returns its ID.

Comments appear in the issue's activity timeline and notify relevant team members.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| issue_id | ID of the issue to comment on | str | Yes |
| comment | Comment text to add to the issue | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| comment_id | ID of the created comment | str |
| comment_body | Text content of the created comment | str |

### Possible use case
<!-- MANUAL: use_case -->
**Automated Updates**: Post status updates or progress reports to issues automatically.

**Integration Comments**: Add comments when external systems (CI/CD, monitoring) detect relevant changes.

**Cross-Tool Communication**: Post comments from chatbots or customer support integrations.
<!-- END MANUAL -->

---
