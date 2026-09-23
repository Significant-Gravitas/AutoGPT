# GitHub Users
<!-- MANUAL: file_description -->
A block for looking up a GitHub user's public profile, or the profile of the account the supplied credentials belong to.
<!-- END MANUAL -->

## Github Get User Info

### What it is
This block fetches information about a GitHub user, or about the authenticated user (yourself) if no username is given.

### How it works
<!-- MANUAL: how_it_works -->
Calls `/users/{username}` when a username is given, and the authenticated-user endpoint `/user` when it is left empty — which is how one block answers both "who is this person" and "who am I". The commonly used fields are pulled out as discrete outputs, and the untouched response is emitted as `user` as well, because GitHub returns roughly forty fields and which ones matter depends on the caller. The two endpoints do not return the same object: for the authenticated user the payload additionally carries private account details such as plan, email and private repository counts, so route `user` with that in mind.
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
| user | The full user object from the API. For the authenticated user (i.e. when no username is given) this also includes private account details such as plan, email and private repository counts. | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
A graph resolves the operating account once at the start of a run — no username, so it hits `/user` — and reuses the returned login to filter issues, pull requests and review requests to that person's own, instead of hardcoding a username into every downstream block.
<!-- END MANUAL -->

---
