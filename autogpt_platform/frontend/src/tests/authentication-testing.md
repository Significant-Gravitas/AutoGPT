# Authentication Testing Documentation

## Overview

The AutoGPT platform frontend uses an automated authentication testing system that creates test users and manages authentication state for all tests. This system ensures reliable authentication testing without manual user management.

## Authentication Flow

### Global User Creation
Before any tests run, the system automatically creates a pool of test users:

- **Timing**: Happens during global setup, before test execution
- **Browser-Specific**: Users are created using the same browser that will run the tests
- **Signup Flow**: Uses the actual application signup process for realistic user creation
- **Storage**: Credentials saved to `.auth/` directory for reuse across test runs

### User Pool Management
- **File Location**: `.auth/user-pool.json` contains all created users
- **Worker Isolation**: Each test worker gets a unique user to prevent conflicts
- **Reuse Strategy**: Existing users are reused when available to speed up test runs
- **Browser Detection**: Uses `BROWSER_TYPE` environment variable to determine which browser to use

## Signin Authentication Tests

### Test Coverage
The signin tests verify core authentication functionality:

- **Login Success**: Users can authenticate with valid credentials
- **Logout Process**: Users can sign out and return to login page
- **Authentication Cycle**: Complete login → logout → login sequence works correctly

### Test Flow
1. Get pre-created test user from the pool
2. Navigate to login page
3. Enter credentials and submit
4. Verify successful authentication (marketplace redirect + profile menu visible)
5. Perform logout if needed
6. Verify logout state (return to login page)

### Authentication Verification
Tests confirm successful authentication by checking:
- **URL Redirect**: User is redirected to `/marketplace` after login
- **UI Elements**: Profile menu becomes visible indicating authenticated state
- **State Persistence**: Authentication state maintains across page interactions

## Signup Authentication Tests

### Test Scenarios
The signup tests cover user creation and immediate authentication:

- **Basic Signup**: Standard user registration flow
- **Custom Credentials**: Signup with specific email/password combinations
- **Form Validation**: Client-side validation rules and error handling
- **Duplicate Users**: Behavior when attempting to register existing email addresses

### Signup Process Testing
1. Navigate to signup page
2. Fill registration form with test data
3. Submit form and handle redirects (onboarding or marketplace)
4. Verify immediate authentication after successful signup
5. Confirm user can access authenticated areas

### Post-Signup Authentication
After successful signup, tests verify:
- **Immediate Login**: User is automatically authenticated
- **Access Rights**: User can access protected marketplace area
- **Profile Availability**: User profile menu is accessible
- **Session State**: Authentication persists for subsequent actions

## Authentication State Management

### User Credentials
- **Generation**: Random email addresses and secure passwords
- **Storage**: Saved in `.auth/` directory (gitignored for security)
- **Reuse**: Same credentials used across multiple test runs
- **Isolation**: Each test worker gets unique user to prevent conflicts

### Session Handling
- **Fresh Sessions**: Each test starts with clean authentication state
- **Login Required**: Tests explicitly authenticate users rather than assuming logged-in state
- **Logout Testing**: Explicit logout verification ensures proper session termination
- **State Verification**: Tests confirm both authenticated and unauthenticated states

## Error Handling

### Signup Failures
- **Server Errors**: Graceful handling when backend is unavailable
- **Network Issues**: Retry logic and timeout handling
- **Validation Errors**: Proper testing of form validation messages
- **Duplicate Emails**: Expected behavior when email already exists

### Signin Failures
- **Invalid Credentials**: Testing of error messages for wrong passwords
- **Account Issues**: Handling of disabled or non-existent accounts
- **Network Problems**: Timeout and connectivity error handling
- **UI Changes**: Flexible selectors that adapt to interface updates

## Local Development

### Setup Requirements
- **Browser Installation**: Playwright browsers via `pnpm playwright install`
- **Environment**: Local database connection for user creation
- **Configuration**: Optional `.env` settings for customization

### Development Benefits
- **Fast Tests**: Pre-created users eliminate signup delays during testing
- **Consistent State**: Each test starts with known authentication state
- **Easy Debugging**: Clear error messages and logging for troubleshooting
- **Flexible Testing**: Easy switching between authenticated and unauthenticated scenarios

## Security Considerations

### Credential Management
- **Random Generation**: Unique email addresses and strong passwords for each user
- **Local Storage**: Credentials stored locally in gitignored directory
- **No Hardcoding**: No test credentials committed to version control
- **Cleanup Options**: Optional removal of test users after test completion

### Authentication Isolation
- **User Separation**: Each test worker uses different credentials
- **Session Independence**: Tests don't share authentication sessions
- **Clean State**: Each test starts with fresh authentication state
- **Conflict Prevention**: No interference between parallel test runs