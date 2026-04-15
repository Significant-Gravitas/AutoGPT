# HubSpot Company
<!-- MANUAL: file_description -->
Blocks for managing HubSpot company records in the CRM.
<!-- END MANUAL -->

## Hub Spot Company

### What it is
Manages HubSpot companies - create, update, and retrieve company information

### How it works
<!-- MANUAL: how_it_works -->
This block interacts with the HubSpot CRM API to manage company records. It supports three operations: create new companies, update existing companies, and retrieve company information by domain.

Company data is passed as a dictionary with standard HubSpot company properties like name, domain, industry, and custom properties.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| operation | Operation to perform (create, update, get) | str | No |
| company_data | Company data for create/update operations | Dict[str, Any] | No |
| domain | Company domain for get/update operations | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| company | Company information | Dict[str, Any] |
| status | Operation status | str |

### Possible use case
<!-- MANUAL: use_case -->
**Lead Enrichment**: Create or update company records when new leads come in from forms or integrations.

**Data Sync**: Keep company information synchronized between HubSpot and other business systems.

**Account Management**: Retrieve company details to personalize communications or trigger workflows.
<!-- END MANUAL -->

---
