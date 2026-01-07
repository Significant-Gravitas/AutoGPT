# Exa Cancel Webset Search

### What it is
Cancel a running webset search.

### What it does
Cancel a running webset search

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| search_id | The ID of the search to cancel | str | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| search_id | The ID of the canceled search | str |
| status | Status after cancellation | str |
| items_found_before_cancel | Number of items found before cancellation | int |
| success | Whether the cancellation was successful | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Exa Create Webset Search

### What it is
Add a new search to an existing webset to find more items.

### What it does
Add a new search to an existing webset to find more items

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| query | Search query describing what to find | str | Yes |
| count | Number of items to find | int | No |
| entity_type | Type of entity to search for | "company" | "person" | "article" | No |
| entity_description | Description for custom entity type | str | No |
| criteria | List of criteria that items must meet. If not provided, auto-detected from query. | List[str] | No |
| behavior | How new results interact with existing items | "override" | "append" | "merge" | No |
| recall | Enable recall estimation for expected results | bool | No |
| exclude_source_ids | IDs of imports/websets to exclude from results | List[str] | No |
| exclude_source_types | Types of sources to exclude ('import' or 'webset') | List[str] | No |
| scope_source_ids | IDs of imports/websets to limit search scope to | List[str] | No |
| scope_source_types | Types of scope sources ('import' or 'webset') | List[str] | No |
| scope_relationships | Relationship definitions for hop searches | List[str] | No |
| scope_relationship_limits | Limits on related entities to find | List[int] | No |
| metadata | Metadata to attach to the search | Dict[str, True] | No |
| wait_for_completion | Wait for the search to complete before returning | bool | No |
| polling_timeout | Maximum time to wait for completion in seconds | int | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| search_id | The unique identifier for the created search | str |
| webset_id | The webset this search belongs to | str |
| status | Current status of the search | str |
| query | The search query | str |
| expected_results | Recall estimation of expected results | Dict[str, True] |
| items_found | Number of items found (if wait_for_completion was True) | int |
| completion_time | Time taken to complete in seconds (if wait_for_completion was True) | float |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Exa Find Or Create Search

### What it is
Find existing search by query or create new - prevents duplicate searches in workflows.

### What it does
Find existing search by query or create new - prevents duplicate searches in workflows

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| query | Search query to find or create | str | Yes |
| count | Number of items to find (only used if creating new search) | int | No |
| entity_type | Entity type (only used if creating) | "company" | "person" | "article" | No |
| behavior | Search behavior (only used if creating) | "override" | "append" | "merge" | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| search_id | The search ID (existing or new) | str |
| webset_id | The webset ID | str |
| status | Current search status | str |
| query | The search query | str |
| was_created | True if search was newly created, False if already existed | bool |
| items_found | Number of items found (0 if still running) | int |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Exa Get Webset Search

### What it is
Get the status and details of a webset search.

### What it does
Get the status and details of a webset search

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| search_id | The ID of the search to retrieve | str | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| search_id | The unique identifier for the search | str |
| status | Current status of the search | str |
| query | The search query | str |
| entity_type | Type of entity being searched | str |
| criteria | Criteria used for verification | List[Dict[str, True]] |
| progress | Search progress information | Dict[str, True] |
| recall | Recall estimation information | Dict[str, True] |
| created_at | When the search was created | str |
| updated_at | When the search was last updated | str |
| canceled_at | When the search was canceled (if applicable) | str |
| canceled_reason | Reason for cancellation (if applicable) | str |
| metadata | Metadata attached to the search | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
