# Exa Websets Search
<!-- MANUAL: file_description -->
Blocks for running and managing searches within Exa websets.
<!-- END MANUAL -->

## Exa Cancel Webset Search

### What it is
Cancel a running webset search

### How it works
<!-- MANUAL: how_it_works -->
This block stops a running search operation on a webset. Items already found before cancellation are retained in the webset.

Use this when a search is taking too long, returning unexpected results, or is no longer needed. The block returns the number of items found before cancellation. It charges only if Exa's response reports a cost (100 credits per USD).
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
**Resource Control**: Stop searches that are taking longer than expected.

**Query Refinement**: Cancel searches to adjust query and restart with better parameters.

**Partial Results**: Stop searches early when you have enough items for your needs.
<!-- END MANUAL -->

---

## Exa Create Webset Search

### What it is
Add a new search to an existing webset to find more items

### How it works
<!-- MANUAL: how_it_works -->
This block adds a search for up to `count` items (1-1000) to an existing webset. `behavior` decides what happens to existing items: 'append' (default) keeps them and adds the new results, 'override' re-evaluates them against this search's criteria and drops the ones that fail; the legacy 'merge' is sent as 'append'. A search can be limited to imports or websets, leave out the items of others, or become a "hop" search through a scope relationship, which needs a limit of 1-10. Before calling Exa the block rejects more than 5 criteria, a 'custom' entity without a 2-200 character description, and source types other than 'import' or 'webset'.

`recall` (on by default) asks Exa to estimate how many matches exist, returned in `expected_results`. With `wait_for_completion` on, the block polls the search (every 5 seconds, backing off to 30) until it is completed, canceled or failed, for up to `polling_timeout` seconds (default 300, max 600); on timeout it returns the status from creation and 0 items instead of failing. It charges only if Exa's response reports a cost (100 credits per USD).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| query | What to find, in plain language (up to 5,000 characters) | str | Yes |
| count | Number of items to find. Exa may find fewer. | int | No |
| entity_type | Kind of result to find; 'auto' lets Exa detect it from the query, 'custom' needs a description | "company" \| "person" \| "article" \| "research_paper" \| "custom" \| "auto" | No |
| entity_description | What each result is, when the entity type is 'custom' (2-200 characters) | str | No |
| criteria | Up to 5 criteria every item must meet. If not provided, auto-detected from query. | List[str] | No |
| behavior | What happens to the webset's existing items: 'override' re-evaluates them against this search's criteria and drops the ones that fail; 'append' keeps them and adds the new results. 'merge' is legacy and is sent as 'append'. | "override" \| "append" \| "merge" | No |
| recall | Also estimate how many matching results exist (returned in 'expected_results') | bool | No |
| max_people_per_company | For people searches: try to include at most this many people from the same current employer (soft cap) | int | No |
| exclude_source_ids | IDs of imports/websets to exclude from results | List[str] | No |
| exclude_source_types | Type of each exclude source, in the same order ('import' or 'webset'; defaults to 'import') | List[str] | No |
| scope_source_ids | IDs of imports/websets to limit search scope to | List[str] | No |
| scope_source_types | Type of each scope source, in the same order ('import' or 'webset'; defaults to 'import') | List[str] | No |
| scope_relationships | Optional, one per scope source: how results relate to that source's items, for hop searches (e.g. 'CEOs of these companies') | List[str] | No |
| scope_relationship_limits | Required for each relationship: related entities to find per source item (1-10) | List[int] | No |
| metadata | Metadata to attach to the search | Dict[str, Any] | No |
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
| expected_results | Recall estimation of expected results | Dict[str, Any] |
| items_found | Number of items found (if wait_for_completion was True) | int |
| completion_time | Time taken to complete in seconds (if wait_for_completion was True) | float |

### Possible use case
<!-- MANUAL: use_case -->
**Webset Expansion**: Add more items to existing websets with new or refined queries.

**Multi-Criteria Collection**: Run multiple searches with different criteria to build comprehensive datasets.

**Iterative Building**: Progressively expand websets based on analysis of initial results.
<!-- END MANUAL -->

---

## Exa Find Or Create Search

### What it is
Find existing search by query or create new - prevents duplicate searches in workflows

### How it works
<!-- MANUAL: how_it_works -->
This block reads the webset and looks for a search whose query matches yours, ignoring case and surrounding spaces. If one exists it returns that search with its items found so far and `was_created` false; otherwise it creates a search with your query, count, entity type and behavior and returns it straight away with 0 items.

The default behavior here is 'override', which re-evaluates the webset's existing items against the new search and drops the ones that fail; choose 'append' to keep them. The 'custom' entity type always fails, because this block has no description input and Exa needs one. It charges only if Exa's response reports a cost (100 credits per USD).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| query | Search query to find or create | str | Yes |
| count | Number of items to find (only used if creating new search) | int | No |
| entity_type | Entity type (only used if creating) | "company" \| "person" \| "article" \| "research_paper" \| "custom" \| "auto" | No |
| behavior | Only used if creating. What happens to the webset's existing items: 'override' re-evaluates them against this search's criteria and drops the ones that fail; 'append' keeps them and adds the new results. 'merge' is legacy and is sent as 'append'. | "override" \| "append" \| "merge" | No |

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
**Retry-Safe Workflows**: Safely handle workflow retries without creating duplicate searches.

**Deduplication**: Avoid running the same search multiple times when called from different workflow branches.

**Efficient Operations**: Skip search creation when results from identical queries already exist.
<!-- END MANUAL -->

---

## Exa Get Webset Search

### What it is
Get the status and details of a webset search

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves one webset search and returns its status, query, entity type, criteria, target count, behavior, scope and exclude sources, progress (found, analyzed, completion, time left) and the recall estimate when one was requested. For a canceled search it also gives the time and reason (webset_deleted, webset_canceled or out_of_credits).

Use this to monitor search progress, verify search configuration, or investigate search behavior when results don't match expectations. It charges only if Exa's response reports a cost (100 credits per USD).
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
| status | Current status: created, pending, running, completed or canceled | str |
| query | The search query | str |
| entity_type | Type of entity being searched | str |
| criteria | Criteria used for verification | List[Dict[str, Any]] |
| count | Number of items the search aims to find | int |
| behavior | How the search treats existing items: override or append | str |
| max_people_per_company | Soft cap on people per employer, if one was requested | int |
| exclude | Imports or websets whose items this search leaves out | List[Dict[str, Any]] |
| scope | Imports or websets the search is limited to (empty for the whole web) | List[Dict[str, Any]] |
| progress | Search progress information | Dict[str, Any] |
| recall | Recall estimation information | Dict[str, Any] |
| created_at | When the search was created | str |
| updated_at | When the search was last updated | str |
| canceled_at | When the search was canceled (if applicable) | str |
| canceled_reason | Why it was canceled: webset_deleted, webset_canceled or out_of_credits | str |
| metadata | Metadata attached to the search | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Progress Tracking**: Monitor search completion and items found during long-running operations.

**Configuration Review**: Retrieve search details to verify criteria and settings are correct.

**Debugging**: Investigate search configuration when results don't match expectations.
<!-- END MANUAL -->

---
