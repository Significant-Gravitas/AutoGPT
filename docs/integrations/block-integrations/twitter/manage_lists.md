# Twitter Manage Lists
<!-- MANUAL: file_description -->
Blocks for creating, updating, and deleting Twitter/X lists.
<!-- END MANUAL -->

## Twitter Create List

### What it is
This block creates a new Twitter List for the authenticated user.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Twitter API v2 via Tweepy to create a new Twitter List under your account. You can specify the list name, description, and whether it should be public or private.

The block authenticates using OAuth 2.0 with list write permissions and sends a POST request to create the list. Returns the new list's ID and URL upon successful creation.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the List to be created | str | No |
| description | Description of the List | str | No |
| private | Whether the List should be private | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| url | URL of the created list | str |
| list_id | ID of the created list | str |

### Possible use case
<!-- MANUAL: use_case -->
**Automated List Setup**: Programmatically create lists as part of onboarding workflows or project initialization.

**Campaign Organization**: Create dedicated lists for tracking accounts related to specific marketing campaigns.

**Research Projects**: Set up new lists to organize accounts for research or monitoring initiatives.
<!-- END MANUAL -->

---

## Twitter Delete List

### What it is
This block deletes a specified Twitter List owned by the authenticated user.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Twitter API v2 via Tweepy to permanently delete a Twitter List that you own. The list and all its member associations are removed and cannot be recovered.

The block authenticates using OAuth 2.0 with list write permissions and sends a DELETE request. Only lists you own can be deleted. Returns a success indicator confirming the deletion.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| list_id | The ID of the List to be deleted | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the deletion was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Account Cleanup**: Remove outdated or unused lists as part of regular account maintenance.

**Project Completion**: Delete temporary lists created for campaigns or projects that have concluded.

**Privacy Management**: Remove lists that are no longer needed to reduce your public profile footprint.
<!-- END MANUAL -->

---

## Twitter Update List

### What it is
This block updates a specified Twitter List owned by the authenticated user.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Twitter API v2 via Tweepy to update the metadata of a Twitter List you own. You can modify the list name and description without affecting the list's members.

The block authenticates using OAuth 2.0 with list write permissions and sends a PUT request with the updated fields. Returns a success indicator confirming the changes were applied.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| list_id | The ID of the List to be updated | str | Yes |
| name | New name for the List | str | No |
| description | New description for the List | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the update was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
**List Rebranding**: Update list names and descriptions to better reflect their evolving purpose.

**Seasonal Updates**: Modify list descriptions to indicate current focus areas or time periods.

**Organization Improvement**: Rename lists to follow consistent naming conventions across your account.
<!-- END MANUAL -->

---
