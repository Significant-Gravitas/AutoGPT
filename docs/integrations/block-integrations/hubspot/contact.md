# HubSpot Contact
<!-- MANUAL: file_description -->
Blocks for managing HubSpot contact records in the CRM.
<!-- END MANUAL -->

## Hub Spot Contact

### What it is
Manages HubSpot contacts - create, update, and retrieve contact information

### How it works
<!-- MANUAL: how_it_works -->
This block interacts with the HubSpot CRM API to manage contact records. It supports creating new contacts, updating existing contacts, and retrieving contacts by email address.

Contact data includes standard properties like email, first name, last name, phone, and any custom properties defined in your HubSpot account.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| operation | Operation to perform (create, update, get) | str | No |
| contact_data | Contact data for create/update operations | Dict[str, Any] | No |
| email | Email address for get/update operations | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| contact | Contact information | Dict[str, Any] |
| status | Operation status | str |

### Possible use case
<!-- MANUAL: use_case -->
**Lead Capture**: Create contacts automatically from form submissions or integrations.

**Contact Updates**: Update contact information when customers change their details.

**CRM Lookup**: Retrieve contact details for personalization or workflow decisions.
<!-- END MANUAL -->

---
