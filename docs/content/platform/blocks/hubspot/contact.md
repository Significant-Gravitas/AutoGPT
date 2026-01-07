# Hub Spot Contact

### What it is
Manages HubSpot contacts - create, update, and retrieve contact information.

### What it does
Manages HubSpot contacts - create, update, and retrieve contact information

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| operation | Operation to perform (create, update, get) | str | No |
| contact_data | Contact data for create/update operations | Dict[str, True] | No |
| email | Email address for get/update operations | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| contact | Contact information | Dict[str, True] |
| status | Operation status | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
