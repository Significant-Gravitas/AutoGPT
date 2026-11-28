# Google Contacts
<!-- MANUAL: file_description -->
Blocks for finding people in the user's Google Contacts and for reading the user's own Google profile. Google Contacts Search needs read access to saved contacts and to "Other contacts" (the `contacts.readonly` and `contacts.other.readonly` scopes). Google Contacts Get My Profile uses the basic profile access every Google connection already has, plus work details (the `user.organization.read` scope).
<!-- END MANUAL -->

## Google Contacts Get My Profile

### What it is
Get the connected Google account's own profile: name, email address, photo, company and job title, plus language when Google shares it.

### How it works
<!-- MANUAL: how_it_works -->
Calls the People API `people.get` endpoint for `people/me`, the signed-in account. It returns the account's name, primary email address and photo, plus company and job title from the user's Google profile (for Google Workspace accounts, also from their directory profile). Name, email and photo come with the basic profile access every Google connection has. Company and job title need the `user.organization.read` scope this block asks for.

Google shares the account language (`locale`) only in some cases, so that output may be missing. `name` and `email` are also output on their own, so you can connect them straight to other blocks.
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| profile | The user's profile | GooglePerson |
| name | The user's display name | str |
| email | The user's primary email address | str |
| locale | The user's language as a BCP 47 tag, such as en-GB, when Google shares it | str |

### Possible use case
<!-- MANUAL: use_case -->
**Email Signatures**: Sign outgoing emails and documents with the user's own name and job title.

**Reports to Self**: Send a daily summary to the connected account's own email address.

**Account Check**: Confirm which Google account an agent is connected to before it changes anything.
<!-- END MANUAL -->

---

## Google Contacts Search

### What it is
Search the user's Google Contacts by name, email address, phone number or company, optionally including 'Other contacts' (people they have emailed but never saved). Returns names, email addresses, phone numbers, companies, job titles and photos.

### How it works
<!-- MANUAL: how_it_works -->
Calls the People API `people.searchContacts` endpoint for saved contacts. When `include_other_contacts` is on (the default), it also calls `otherContacts.search` for people the user has emailed but never saved. Google answers these searches from a cache that it refreshes after each request. So the block first sends each search with an empty query, as Google recommends, to start a refresh. Then it runs the real search. If no one is found at all, it repeats the searches once, about 2 seconds later. A contact added or changed in the last few minutes may still be missing.

Saved contacts come first. An other contact is dropped when it shares an email address or phone number with a saved contact, and the list is cut to `max_results` (Google returns at most 30 per source). Other contacts carry only a name, email addresses and phone numbers. The block always asks for both scopes, `contacts.readonly` and `contacts.other.readonly`, even with the toggle off, because a block's Google scopes are fixed.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Name, email address, phone number or company to look for. Matches the start of words, so 'Ali' finds 'Alice Smith'. | str | Yes |
| include_other_contacts | Also search 'Other contacts': people the user has emailed but never saved. These have only a name, email and phone number. | bool | No |
| max_results | Maximum number of people to return (Google allows up to 30) | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| contacts | Matching people, saved contacts first | List[GooglePerson] |
| contact | Each matching person | GooglePerson |

### Possible use case
<!-- MANUAL: use_case -->
**Email Someone by Name**: Look up "Sarah from Acme" and pass her email address to Gmail Send once the user confirms it's the right person.

**Lead Enrichment**: Check whether an email sender is already a contact and pull their company and job title.

**Meeting Prep**: Find the phone number and company of each person on a calendar invite.
<!-- END MANUAL -->

---
