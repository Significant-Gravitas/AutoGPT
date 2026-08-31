# GitHub Users
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Github Get User Info

### What it is
This block fetches information about a GitHub user, or about the authenticated user (yourself) if no username is given.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| username | Username of the GitHub user to look up. Leave empty to get the authenticated user (yourself). | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if fetching the user info failed | str |
| username | Login (username) of the user | str |
| name | Display name of the user | str |
| profile_url | URL of the user's GitHub profile | str |
| avatar_url | URL of the user's avatar image | str |
| user | The full user object from the API | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
