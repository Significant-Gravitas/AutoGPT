# HubSpot Deal
<!-- MANUAL: file_description -->
Blocks for creating, finding and linking HubSpot deals, and for listing the deal pipelines and their stages.
<!-- END MANUAL -->

## Hub Spot Associate Deal

### What it is
Associates a HubSpot deal with a contact or a company

### How it works
<!-- MANUAL: how_it_works -->
The block creates HubSpot's default association between a deal and a contact or a company, through the v4 associations API. HubSpot picks the association type itself (deal to contact, or deal to company), so no type ID is needed.

If HubSpot reports an error for the association, the block fails with HubSpot's message instead of reporting success.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| deal_id | ID of the deal | str | Yes |
| object_type | What to associate the deal with | "contact" \| "company" | No |
| object_id | ID of the contact or company | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| association | HubSpot's report of the association it made | Dict[str, Any] |
| status | Operation status | str |

### Possible use case
<!-- MANUAL: use_case -->
**Link a New Deal**: Attach the contact and company a deal was opened for, right after creating it.

**Lead Conversion**: When a lead qualifies, open a deal and associate it with that lead's contact record.

**Account Roll-up**: Associate deals with the company they belong to, so the company page in HubSpot shows its whole pipeline.
<!-- END MANUAL -->

---

## Hub Spot Deal

### What it is
Creates or updates a HubSpot deal

### How it works
<!-- MANUAL: how_it_works -->
With `operation` set to `create`, the block creates a deal from `deal_data`, a dictionary of HubSpot deal properties by their internal names, such as `dealname`, `amount`, `dealstage`, `pipeline` and `closedate`. With `update`, it changes those properties on the deal whose ID is `deal_id`, and fails without calling HubSpot when no ID is given.

`dealstage` and `pipeline` take HubSpot's internal IDs, not their labels; the HubSpot Deal Pipelines block lists them. HubSpot recommends setting `dealname` and `dealstage` on a new deal, and uses the default pipeline when none is given.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| operation | Operation to perform (create, update) | "create" \| "update" | No |
| deal_data | Deal properties by internal name, e.g. dealname, amount, dealstage, pipeline, closedate. Stages and pipelines are set by ID. | Dict[str, Any] | No |
| deal_id | ID of the deal to update | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| deal | The deal as HubSpot returns it | Dict[str, Any] |
| status | Operation status | str |

### Possible use case
<!-- MANUAL: use_case -->
**Open Deals from Qualified Leads**: Create a deal with an amount and a close date when a lead replies or books a meeting.

**Move Deals Along**: Update a deal's stage when a proposal is sent or a contract is signed elsewhere.

**Keep Forecasts Current**: Write a new amount or close date onto a deal when the customer's plans change.
<!-- END MANUAL -->

---

## Hub Spot Deal Pipelines

### What it is
Lists the HubSpot deal pipelines and their stages

### How it works
<!-- MANUAL: how_it_works -->
The block reads every deal pipeline in the HubSpot account, with its stages. `pipelines` is HubSpot's own response, and `stages` lists each stage of each pipeline as one row: the pipeline's ID and label, the stage's ID, label and display order, its win probability from 0 to 1, and whether it is a closed stage.

HubSpot stores a deal's stage and pipeline as IDs, so this is the list that turns them into names, or a name into the ID to filter or update by.
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| pipelines | Each deal pipeline with its stages, as HubSpot returns them | List[Dict[str, Any]] |
| stages | Every stage of every pipeline, one row each; probability runs from 0 to 1 | List[HubSpotDealStage] |
| status | Operation status | str |

### Possible use case
<!-- MANUAL: use_case -->
**Pipeline Boards**: Lay out a sales page's columns in each pipeline's stage order, with HubSpot's own labels.

**Weighted Pipeline**: Multiply each open deal's amount by its stage's probability to estimate what the pipeline is worth.

**Stage Lookups**: Find the ID of the "Closed won" stage before searching for or updating deals in it.
<!-- END MANUAL -->

---

## Hub Spot Get Deal

### What it is
Gets a HubSpot deal by ID, with its contacts and companies

### How it works
<!-- MANUAL: how_it_works -->
The block fetches one deal by its HubSpot ID, with the properties named in `properties`, and asks HubSpot for the contacts and companies associated with it. `contact_ids` and `company_ids` list each associated record once, even when HubSpot links it under more than one association type.

A property name HubSpot does not know is left out of the deal rather than causing an error, and a defined property with no value comes back as `null`.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| deal_id | ID of the deal | str | Yes |
| properties | Deal properties to return. HubSpot leaves out a name it does not know. | List[str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| deal | The deal as HubSpot returns it | Dict[str, Any] |
| contact_ids | IDs of the contacts on the deal | List[str] |
| company_ids | IDs of the companies on the deal | List[str] |
| status | Operation status | str |

### Possible use case
<!-- MANUAL: use_case -->
**Deal Detail Pages**: Show a deal's amount, stage and close date next to the people and company on it.

**Follow-up Drafting**: Fetch a deal and its contacts so a workflow can write to the right people about it.

**Data Checks**: Read a deal before updating it, to act only when its stage or amount is what the workflow expects.
<!-- END MANUAL -->

---

## Hub Spot Search Deals

### What it is
Searches HubSpot deals by text, pipeline, stage or any deal property

### How it works
<!-- MANUAL: how_it_works -->
The block runs HubSpot's CRM search over deals. `query` searches the deal name, pipeline, stage, description and type; `pipeline_id`, `stage_id` and any extra `filters` are combined so that a deal must match all of them. Results can be sorted by one property, and `properties` picks which properties each deal carries.

HubSpot returns at most 200 deals per page and 10,000 per search. When there are more, pass `next_after` back in as `after` to get the next page; it is empty on the last page.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Text to find in the deal name, pipeline, stage, description or type | str | No |
| pipeline_id | Only deals in this pipeline | str | No |
| stage_id | Only deals in this stage | str | No |
| filters | More HubSpot filters, all of which must match, e.g. {"propertyName": "amount", "operator": "GT", "value": "1000"} | List[Dict[str, Any]] | No |
| sort_by | Deal property to sort by, e.g. closedate | str | No |
| sort_direction | Sort order when sort_by is set | "ASCENDING" \| "DESCENDING" | No |
| properties | Deal properties to return | List[str] | No |
| limit | Deals per page, at most 200 | int | No |
| after | next_after from the previous page, to get the next one | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| deals | Matching deals, as HubSpot returns them | List[Dict[str, Any]] |
| total | How many deals match in all | int |
| next_after | Cursor for the next page; empty on the last page | str |
| status | Operation status | str |

### Possible use case
<!-- MANUAL: use_case -->
**Pipeline Views**: List every open deal in a pipeline and stage for a sales page or a weekly report.

**Closing Soon**: Find deals whose close date falls this month, sorted by close date, to chase before the month ends.

**Stalled Deals**: Find deals not modified in weeks, so a workflow can flag them or draft a nudge.
<!-- END MANUAL -->

---
