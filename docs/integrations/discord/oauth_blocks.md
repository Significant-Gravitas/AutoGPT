# Discord OAuth Blocks
<!-- MANUAL: file_description -->
Blocks for Discord OAuth2 authentication and retrieving user information.
<!-- END MANUAL -->

## Discord Get Current User

### What it is
Gets information about the currently authenticated Discord user using OAuth2 credentials.

### How it works
<!-- MANUAL: how_it_works -->
This block uses Discord's OAuth2 API to retrieve information about the currently authenticated user. It requires valid OAuth2 credentials that have been obtained through Discord's authorization flow with the `identify` scope.

The block queries the Discord `/users/@me` endpoint and returns the user's profile information including their unique ID, username, avatar, and customization settings like banner and accent color.
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| user_id | The authenticated user's Discord ID | str |
| username | The user's username | str |
| avatar_url | URL to the user's avatar image | str |
| banner_url | URL to the user's banner image (if set) | str |
| accent_color | The user's accent color as an integer | int |

### Possible use case
<!-- MANUAL: use_case -->
**User Authentication**: Verify user identity after OAuth login to personalize experiences or grant access.

**Profile Integration**: Display Discord user information in external applications or dashboards.

**Account Linking**: Connect Discord accounts with other services using the unique user ID.
<!-- END MANUAL -->

---
