# Google Contacts Directory
<!-- MANUAL: file_description -->
A block for searching the user's Google Workspace directory: colleagues' profiles and the contacts the Workspace admin shared with the whole organization. It needs read access to the directory (the `directory.readonly` scope) and only works for Google Workspace (work or school) accounts. Personal Google accounts have no directory.
<!-- END MANUAL -->

## Google Contacts Search Directory

### What it is
Search the user's Google Workspace directory for colleagues and contacts shared with the organization, by name or email address. Returns names, email addresses, phone numbers, job titles and photos. Only works for Google Workspace (work or school) accounts.

### How it works
<!-- MANUAL: how_it_works -->
Calls the People API `people.searchDirectoryPeople` endpoint, which matches the start of words in names, email addresses and other directory fields. It searches colleagues' directory profiles and, unless you turn off `include_shared_contacts`, the organization's shared contacts too. Each result has the person's name, email addresses, phone numbers, job title and photo. Use `next_page_token` to page through large result sets (up to 500 people per page).

A personal Google account gets an error that points to Google Contacts Search instead. What the search can see depends on the organization's directory settings: Google only returns directory data when the admin allows contact sharing.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Name or email address to look for. Matches the start of words, so 'Ali' finds 'Alice Smith'. | str | Yes |
| include_shared_contacts | Also search contacts the Workspace admin shared with the whole organization, not just colleagues' directory profiles | bool | No |
| max_results | Maximum number of people to return | int | No |
| page_token | Page token from a previous search, to get the next page | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| people | Matching colleagues and shared contacts | List[GooglePerson] |
| person | Each matching person | GooglePerson |
| next_page_token | Token for the next page, when there are more results | str |

### Possible use case
<!-- MANUAL: use_case -->
**Find a Colleague's Email**: Turn "Priya in finance" into an email address before inviting her to a meeting.

**Introductions**: Get a coworker's job title and phone number for an introduction email.

**Shared Vendor Contacts**: Find a supplier contact that the admin shared with the whole company.
<!-- END MANUAL -->

---
