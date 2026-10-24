# Linear Issue Lifecycle
<!-- MANUAL: file_description -->
Archive completed work or move unwanted issues to Linear's trash.
<!-- END MANUAL -->

## Linear Archive Issue

### What it is
Archives a Linear issue, removing it from active views while retaining it in the archive.

### How it works
<!-- MANUAL: how_it_works -->
Calls Linear's `issueArchive` mutation for the supplied issue UUID or identifier. The block emits success only after Linear confirms the archive. OAuth credentials require `write` access, and sensitive-action review applies when safe mode is enabled.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| issue_id | Issue UUID or identifier, such as ENG-123. | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| issue_id | ID or identifier of the affected issue. | str |
| success | Whether Linear confirmed the operation. | bool |

### Possible use case
<!-- MANUAL: use_case -->
Archive issues after a workflow verifies that their work is complete.
<!-- END MANUAL -->

---

## Linear Delete Issue

### What it is
Deletes a Linear issue using Linear's recoverable trash behavior. Does not request immediate permanent deletion.

### How it works
<!-- MANUAL: how_it_works -->
Calls Linear's `issueDelete` mutation, which moves the issue to trash using Linear's normal recovery behavior. The block has no permanent-deletion option. It emits success only after Linear confirms the deletion. OAuth credentials require `write` access, and sensitive-action review applies when safe mode is enabled.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| issue_id | Issue UUID or identifier, such as ENG-123. | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| issue_id | ID or identifier of the affected issue. | str |
| success | Whether Linear confirmed the operation. | bool |

### Possible use case
<!-- MANUAL: use_case -->
Move duplicate or accidentally created issues to trash after identifying the issue to keep.
<!-- END MANUAL -->

---
