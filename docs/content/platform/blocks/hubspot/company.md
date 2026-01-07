# Hub Spot Company

### What it is
Manages HubSpot companies - create, update, and retrieve company information.

### What it does
Manages HubSpot companies - create, update, and retrieve company information

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| operation | Operation to perform (create, update, get) | str | No |
| company_data | Company data for create/update operations | Dict[str, True] | No |
| domain | Company domain for get/update operations | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| company | Company information | Dict[str, True] |
| status | Operation status | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
