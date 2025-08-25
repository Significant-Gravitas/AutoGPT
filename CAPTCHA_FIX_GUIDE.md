# CAPTCHA Display Issue Fix Guide

## Root Cause Analysis

The CAPTCHA is not displaying because the application is configured to bypass CAPTCHA verification entirely. Users see the message "please complete captcha first" but no CAPTCHA widget appears.

## Issues Found

### 1. Frontend Configuration Issues
- `NEXT_PUBLIC_BEHAVE_AS=LOCAL` in `.env.default` → Should be `CLOUD` for CAPTCHA to show
- `NEXT_PUBLIC_TURNSTILE=disabled` in `.env.default` → Should be `enabled`
- Missing `NEXT_PUBLIC_CLOUDFLARE_TURNSTILE_SITE_KEY` → Required for CAPTCHA widget

### 2. Backend Configuration Issues
- Missing `TURNSTILE_SECRET_KEY` → Required for server-side token verification

## How CAPTCHA Logic Works

### Frontend Logic (useTurnstile.ts)
```typescript
// CAPTCHA only renders when ALL conditions are true:
setBearerToken(
  behaveAs === BehaveAs.CLOUD &&      // Must be CLOUD mode
  hasTurnstileKey &&                  // Must have site key
  !turnstileDisabled                  // Must not be disabled
);

// If any condition fails, CAPTCHA is bypassed:
if (turnstileDisabled || behaveAs !== BehaveAs.CLOUD || !hasTurnstileKey) {
  setVerified(true); // Bypasses CAPTCHA validation
}
```

### UI Display Logic (signup/page.tsx)
```tsx
{/* CAPTCHA only shows when environment is cloud AND not verified */}
{isCloudEnv && !turnstile.verified ? (
  <Turnstile ... />
) : null}
```

## Fix Instructions

### For Production/Cloud Environment

1. **Frontend Environment Variables** (`.env` or `.env.local`):
```bash
# Set behavior to cloud mode
NEXT_PUBLIC_BEHAVE_AS=CLOUD

# Enable Turnstile
NEXT_PUBLIC_TURNSTILE=enabled

# Add your Cloudflare Turnstile site key
NEXT_PUBLIC_CLOUDFLARE_TURNSTILE_SITE_KEY=your_site_key_here
```

2. **Backend Environment Variables** (`.env` or environment config):
```bash
# Add your Cloudflare Turnstile secret key
TURNSTILE_SECRET_KEY=your_secret_key_here
```

3. **Get Cloudflare Turnstile Keys**:
   - Go to [Cloudflare Dashboard](https://dash.cloudflare.com/)
   - Navigate to Turnstile
   - Create a new site or use existing
   - Copy Site Key (public) and Secret Key (private)

### For Development/Testing Environment

If you want to **disable** CAPTCHA intentionally:
```bash
# Keep as LOCAL to bypass CAPTCHA
NEXT_PUBLIC_BEHAVE_AS=LOCAL
NEXT_PUBLIC_TURNSTILE=disabled
```

### Immediate Fix for Current Users

If you need to quickly fix the issue without setting up Cloudflare Turnstile:

**Option 1: Disable CAPTCHA requirement temporarily**
- Remove CAPTCHA validation from signup/login forms
- Set `NEXT_PUBLIC_TURNSTILE=disabled` and `NEXT_PUBLIC_BEHAVE_AS=LOCAL`

**Option 2: Show proper error message**
- Update the UI to show "CAPTCHA is currently unavailable" instead of "please complete captcha first"
- Allow users to proceed without CAPTCHA when it's not configured

## Files Involved

### Frontend Files
- `autogpt_platform/frontend/src/hooks/useTurnstile.ts` - CAPTCHA logic
- `autogpt_platform/frontend/src/app/(platform)/signup/page.tsx` - Signup page
- `autogpt_platform/frontend/src/app/(platform)/login/page.tsx` - Login page  
- `autogpt_platform/frontend/src/app/(platform)/reset-password/page.tsx` - Reset password page
- `autogpt_platform/frontend/src/components/auth/Turnstile.tsx` - CAPTCHA widget component

### Backend Files
- `autogpt_platform/backend/backend/server/v2/turnstile/routes.py` - Verification endpoint
- `autogpt_platform/backend/backend/util/settings.py` - Configuration settings

## Testing the Fix

1. Set the environment variables as described above
2. Restart frontend and backend services
3. Navigate to signup/login page
4. Verify CAPTCHA widget appears
5. Complete CAPTCHA and verify form submission works

## Additional Improvements Recommended

1. Add better error handling when CAPTCHA fails to load
2. Show user-friendly messages when CAPTCHA is misconfigured
3. Add health check endpoint for CAPTCHA configuration
4. Consider adding retry mechanism for failed CAPTCHA verifications