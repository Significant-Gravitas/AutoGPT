# Linear OAuth Refresh Token Implementation

## Overview

This document describes the implementation of Linear's new OAuth2 refresh token system in the AutoGPT platform. Linear is migrating from long-lived access tokens to short-lived access tokens with refresh tokens, with full transition required by April 1, 2026.

## Changes Made

### 1. Updated Linear OAuth Handler (`/autogpt_platform/backend/backend/blocks/linear/_oauth.py`)

#### Key Improvements:

1. **Refresh Token Support**: 
   - Added support for refresh tokens in token requests
   - Implemented HTTP Basic Authentication for refresh token requests (preferred by Linear)
   - Enhanced error handling with detailed error descriptions

2. **Token Migration**:
   - Added `migrate_old_token()` method to migrate old long-lived tokens to new short-lived tokens with refresh tokens
   - Added `needs_migration()` method to detect old tokens that need migration
   - Overridden `get_access_token()` to automatically handle migration when needed

3. **Enhanced Token Handling**:
   - Updated `_request_tokens()` to handle both authorization code and refresh token flows
   - Added proper token expiration handling for short-lived tokens
   - Improved username extraction and preservation during token refresh

4. **Backward Compatibility**:
   - Maintains support for existing long-lived tokens
   - Graceful fallback if migration fails
   - Automatic detection of token types

#### New Methods:

- `migrate_old_token(credentials)`: Migrates old tokens using Linear's `/oauth/migrate_old_token` endpoint
- `needs_migration(credentials)`: Checks if credentials need migration
- `get_access_token(credentials)`: Enhanced method with automatic migration support

### 2. Updated Test Credentials (`/autogpt_platform/backend/backend/blocks/linear/_config.py`)

- Updated mock credentials to include proper expiration times for testing short-lived tokens
- Added realistic scope examples

## Technical Details

### Authentication Methods

The implementation supports both authentication methods recommended by Linear:

1. **HTTP Basic Authentication** (preferred for refresh tokens):
   ```http
   Authorization: Basic <base64(client_id:client_secret)>
   ```

2. **Form Parameters**:
   ```http
   client_id=...&client_secret=...
   ```

### Token Flow

1. **New Tokens**: Receive both access and refresh tokens with expiration
2. **Old Token Migration**: Automatically detects and migrates old tokens
3. **Token Refresh**: Uses refresh tokens to get new access tokens when needed
4. **Fallback**: Maintains backward compatibility with old long-lived tokens

### Error Handling

- Enhanced error messages with both error and error_description fields
- Graceful handling of migration failures
- Proper exception types for different failure scenarios

## Migration Strategy

The implementation provides a seamless migration path:

1. **Automatic Detection**: Old tokens are detected by having no expiration and no refresh token
2. **On-Demand Migration**: When `get_access_token()` is called on old tokens, migration is attempted
3. **Fallback**: If migration fails, the old token is used as-is for backward compatibility
4. **Manual Migration**: The `migrate_old_token()` method can be called explicitly if needed

## Testing

The implementation has been tested for:

- ✅ OAuth URL generation with proper parameter encoding
- ✅ HTTP Basic Authentication header generation
- ✅ Token expiration logic with 5-minute buffer
- ✅ Migration detection for old vs new tokens
- ✅ Syntax and import compatibility

## Next Steps

1. **Enable Refresh Tokens in Linear Console**: Navigate to Linear OAuth application settings and enable refresh tokens
2. **Monitor Token Usage**: Watch for automatic migrations in logs
3. **Update Credentials Store**: Consider implementing automatic credential updates when migration occurs
4. **Testing**: Test with real Linear OAuth flow once refresh tokens are enabled

## Important Dates

- **October 1, 2025**: New OAuth applications use refresh tokens by default
- **April 1, 2026**: All newly issued tokens will be short-lived (deadline for migration)

## References

- [Linear OAuth Documentation](https://linear.app/developers/oauth-2-0-authentication)
- [Linear Token Refresh Guide](https://linear.app/developers/oauth-2-0-authentication#refresh-an-access-token)
- [Linear Migration Guide](https://linear.app/developers/oauth-2-0-authentication#migrate-to-using-refresh-tokens)