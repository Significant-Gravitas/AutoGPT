# Linear Issue Update
<!-- MANUAL: file_description -->
Update existing Linear issues from an agent workflow.
<!-- END MANUAL -->

## Linear Update Issue

### What it is
Updates a Linear issue's title, description, status, priority, assignee, labels, due date, or estimate.

### How it works
<!-- MANUAL: how_it_works -->
Provide an issue UUID or identifier and a `changes` object. Omitted or null values leave fields unchanged. Set `clear_fields` to `assignee`, `description`, `due_date`, or `estimate` to explicitly clear them. An empty description or `label_ids: []` also clears that field.

Use workflow-state UUIDs from the issue's team for `state_id`, user UUIDs for `assignee_id`, and label UUIDs for labels. `add_label_ids` and `remove_label_ids` change labels atomically without replacing unrelated labels. Do not combine them with `label_ids` or add and remove the same label.

`priority: 0` removes the priority. `due_date` uses `YYYY-MM-DD`; `estimate` is an integer supported by the team. Invalid or empty change requests are rejected before calling Linear.

OAuth credentials require `write` access. This block participates in sensitive-action review when safe mode is enabled.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| issue_id | Issue UUID or identifier, such as ENG-123. | str | Yes |
| changes | Fields to change. Omitted values leave existing fields unchanged; use clear_fields to clear nullable fields. | Dict[str, Any] | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| issue | The updated Linear issue. | Issue |

### Possible use case
<!-- MANUAL: use_case -->
Move an issue to the team's review state after a pull request opens, assign it to a reviewer, and add a review label without removing existing labels.
<!-- END MANUAL -->

---
