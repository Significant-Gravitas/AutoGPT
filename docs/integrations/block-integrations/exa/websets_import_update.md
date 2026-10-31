# Exa Websets Import Update
<!-- MANUAL: file_description -->
Blocks for renaming Exa Websets imports and replacing their metadata.
<!-- END MANUAL -->

## Exa Update Import

### What it is
Update the title or metadata of an Exa Websets import

### How it works
<!-- MANUAL: how_it_works -->
This block sends a PATCH to the Websets API's `/v0/imports/{id}` with only the fields you set, so anything left empty stays as it is. Metadata is replaced as a whole, not merged. The imported data itself can't be changed after upload; create a new import for that.

If neither a title nor metadata is set, the block fails before calling Exa. It returns the import's status, title, metadata and update time as Exa reports them after the change. It charges only if Exa's response reports a cost (100 credits per USD).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| import_id | The ID of the import to update | str | Yes |
| title | New title for the import. Leave empty to keep the current one. | str | No |
| metadata | Key-value pairs that replace the import's metadata. Leave empty to keep the current metadata. | Dict[str, str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| import_id | The ID of the updated import | str |
| status | Import status: pending, processing, completed, failed or canceled | str |
| title | The import's title after the update | str |
| metadata | The import's metadata after the update | Dict[str, Any] |
| updated_at | When the import was last updated | str |

### Possible use case
<!-- MANUAL: use_case -->
**Clear Naming**: Rename an import after upload so it is easy to find when attaching it to websets.

**Provenance Tracking**: Store the source system and upload date in the import's metadata.

**Status Check After Edit**: Confirm an import is completed while updating its title in the same step.
<!-- END MANUAL -->

---
