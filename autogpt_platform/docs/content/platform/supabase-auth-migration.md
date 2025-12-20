# Migrating from Supabase Auth to Native FastAPI Auth

This guide covers the complete migration from Supabase Auth to native FastAPI authentication.

## Overview

The migration replaces Supabase Auth with a native FastAPI implementation while:
- Maintaining the same JWT format so existing sessions remain valid
- Keeping the frontend interface identical (no component changes needed)
- Supporting both password and Google OAuth authentication
- Providing admin impersonation and user management

## Prerequisites

- Access to the production database
- Postmark API credentials configured
- Google OAuth credentials (if using Google sign-in)
- Ability to deploy backend and frontend changes

---

## Phase 1: Backend Setup

### 1.1 Install Dependencies

```bash
cd autogpt_platform/backend
poetry add argon2-cffi
```

### 1.2 Run Database Migration

Create and apply the Prisma migration:

```bash
cd autogpt_platform/backend
poetry run prisma migrate dev --name add_native_auth
```

This adds:
- `passwordHash`, `authProvider`, `migratedFromSupabase`, `emailVerifiedAt` fields to `User`
- `UserAuthRefreshToken` table for session management
- `UserAuthMagicLink` table for email verification and password reset

### 1.3 Configure Environment Variables

Add to your `.env` file:

```bash
# Admin Configuration
ADMIN_EMAIL_DOMAINS=agpt.co,autogpt.com
ADMIN_EMAILS=specific-admin@example.com

# Google OAuth (if using)
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_REDIRECT_URI=https://your-domain.com/api/auth/oauth/google/callback

# Frontend URL for redirects
FRONTEND_BASE_URL=https://your-domain.com

# Postmark (should already be configured)
POSTMARK_SERVER_API_TOKEN=your-postmark-token
POSTMARK_SENDER_EMAIL=noreply@your-domain.com
```

### 1.4 Deploy Backend

Deploy the backend with the new auth endpoints. The new endpoints are:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/signup` | POST | Register new user |
| `/api/auth/login` | POST | Login with email/password |
| `/api/auth/logout` | POST | Logout |
| `/api/auth/refresh` | POST | Refresh access token |
| `/api/auth/me` | GET | Get current user |
| `/api/auth/password/reset` | POST | Request password reset |
| `/api/auth/password/set` | POST | Set new password |
| `/api/auth/verify-email` | GET | Verify email from link |
| `/api/auth/oauth/google/authorize` | GET | Start Google OAuth |
| `/api/auth/oauth/google/callback` | GET | Google OAuth callback |

---

## Phase 2: User Migration

### 2.1 Check Migration Status

```bash
cd autogpt_platform/backend
poetry run python -m backend.data.auth.migration --status
```

This shows:
```
Migration Status:
----------------------------------------
Total users: 10000
Already using native auth: 0
OAuth users (Google): 1500
Migrated, pending password: 0
Not yet migrated: 8500
```

### 2.2 Generate Pre-Migration Report

```bash
poetry run python -m backend.data.auth.migration --report
```

This creates a CSV file with all users and their current status.

### 2.3 Dry Run

Test the migration without making changes:

```bash
poetry run python -m backend.data.auth.migration --dry-run --full-migration
```

### 2.4 Run Migration (Mark Users)

Mark all existing Supabase users as migrated:

```bash
poetry run python -m backend.data.auth.migration --mark-migrated --batch-size 500
```

### 2.5 Send Password Reset Emails

Send emails to users who need to set their password:

```bash
# Start with a small batch to verify emails work
poetry run python -m backend.data.auth.migration --send-emails --batch-size 10

# Then send to everyone
poetry run python -m backend.data.auth.migration --send-emails --batch-size 100 --email-delay 0.5
```

**Note:** OAuth users (Google) are automatically skipped - they continue using Google sign-in.

---

## Phase 3: Frontend Migration

The frontend uses a Supabase client abstraction layer. We need to replace the internals while keeping the interface identical.

### 3.1 Understanding the Architecture

The frontend has these Supabase-related files:

```
src/lib/supabase/
├── actions.ts              # Server actions (validateSession, logout, etc.)
├── middleware.ts           # Next.js middleware for session validation
├── helpers.ts              # Utility functions
├── server/
│   └── getServerSupabase.ts  # Server-side Supabase client
└── hooks/
    ├── helpers.ts          # Client-side helpers
    ├── useSupabase.ts      # Main auth hook
    └── useSupabaseStore.ts # Zustand store for auth state
```

### 3.2 Option A: Gradual Migration (Recommended)

Keep Supabase running during migration and gradually switch endpoints.

#### Step 1: Create Native Auth Client

The native auth client is already created at `src/lib/auth/native-auth.ts`. It provides:

```typescript
// Client-side functions
getAccessToken()           // Get token from cookie
isAuthenticated()          // Check if user is authenticated
getCurrentUserFromToken()  // Parse user from JWT

// Server-side functions (for server actions)
serverLogin(email, password)
serverSignup(email, password)
serverLogout(scope)
serverRefreshToken()
serverGetCurrentUser()
serverRequestPasswordReset(email)
serverSetPassword(token, password)
serverGetGoogleAuthUrl(redirectTo)
```

#### Step 2: Update Login Action

Edit `src/app/(platform)/login/actions.ts`:

```typescript
"use server";

import { serverLogin } from "@/lib/auth/native-auth";
import { loginFormSchema } from "@/types/auth";
import * as Sentry from "@sentry/nextjs";
import BackendAPI from "@/lib/autogpt-server-api";
import { shouldShowOnboarding } from "../../api/helpers";

export async function login(email: string, password: string) {
  try {
    const parsed = loginFormSchema.safeParse({ email, password });

    if (!parsed.success) {
      return {
        success: false,
        error: "Invalid email or password",
      };
    }

    const result = await serverLogin(parsed.data.email, parsed.data.password);

    if (!result.success) {
      return {
        success: false,
        error: result.error || "Login failed",
      };
    }

    // Create user in backend if needed
    const api = new BackendAPI();
    await api.createUser();

    const onboarding = await shouldShowOnboarding();

    return {
      success: true,
      onboarding,
    };
  } catch (err) {
    Sentry.captureException(err);
    return {
      success: false,
      error: "Failed to login. Please try again.",
    };
  }
}
```

#### Step 3: Update Signup Action

Edit `src/app/(platform)/signup/actions.ts`:

```typescript
"use server";

import { serverSignup } from "@/lib/auth/native-auth";
import { signupFormSchema } from "@/types/auth";
import * as Sentry from "@sentry/nextjs";

export async function signup(
  email: string,
  password: string,
  confirmPassword: string,
  agreeToTerms: boolean,
) {
  try {
    const parsed = signupFormSchema.safeParse({
      email,
      password,
      confirmPassword,
      agreeToTerms,
    });

    if (!parsed.success) {
      return {
        success: false,
        error: "Invalid signup payload",
      };
    }

    const result = await serverSignup(parsed.data.email, parsed.data.password);

    if (!result.success) {
      if (result.error === "Email already registered") {
        return { success: false, error: "user_already_exists" };
      }
      return {
        success: false,
        error: result.error || "Signup failed",
      };
    }

    // User needs to verify email before logging in
    return {
      success: true,
      message: result.message,
      requiresVerification: true,
    };
  } catch (err) {
    Sentry.captureException(err);
    return {
      success: false,
      error: "Failed to sign up. Please try again.",
    };
  }
}
```

#### Step 4: Update Server Actions

Edit `src/lib/supabase/actions.ts`:

```typescript
"use server";

import * as Sentry from "@sentry/nextjs";
import { revalidatePath } from "next/cache";
import { cookies } from "next/headers";
import { getRedirectPath } from "./helpers";
import {
  serverGetCurrentUser,
  serverLogout as nativeLogout,
  serverRefreshToken,
} from "@/lib/auth/native-auth";

// User type compatible with existing code
interface User {
  id: string;
  email: string;
  role?: string;
}

export interface SessionValidationResult {
  user: User | null;
  isValid: boolean;
  redirectPath?: string;
}

export async function validateSession(
  currentPath: string,
): Promise<SessionValidationResult> {
  return await Sentry.withServerActionInstrumentation(
    "validateSession",
    {},
    async () => {
      try {
        const { user, error } = await serverGetCurrentUser();

        if (error || !user) {
          const redirectPath = getRedirectPath(currentPath);
          return {
            user: null,
            isValid: false,
            redirectPath: redirectPath || undefined,
          };
        }

        return {
          user: {
            id: user.id,
            email: user.email,
            role: user.role,
          },
          isValid: true,
        };
      } catch (error) {
        console.error("Session validation error:", error);
        const redirectPath = getRedirectPath(currentPath);
        return {
          user: null,
          isValid: false,
          redirectPath: redirectPath || undefined,
        };
      }
    },
  );
}

export async function getCurrentUser(): Promise<{
  user: User | null;
  error?: string;
}> {
  return await Sentry.withServerActionInstrumentation(
    "getCurrentUser",
    {},
    async () => {
      try {
        const { user, error } = await serverGetCurrentUser();

        if (error) {
          return { user: null, error };
        }

        if (!user) {
          return { user: null };
        }

        return {
          user: {
            id: user.id,
            email: user.email,
            role: user.role,
          }
        };
      } catch (error) {
        console.error("Get current user error:", error);
        return {
          user: null,
          error: error instanceof Error ? error.message : "Unknown error",
        };
      }
    },
  );
}

export async function getWebSocketToken(): Promise<{
  token: string | null;
  error?: string;
}> {
  return await Sentry.withServerActionInstrumentation(
    "getWebSocketToken",
    {},
    async () => {
      try {
        // Get access token from cookie
        const cookieStore = await cookies();
        const token = cookieStore.get("access_token")?.value;
        return { token: token || null };
      } catch (error) {
        console.error("Get WebSocket token error:", error);
        return {
          token: null,
          error: error instanceof Error ? error.message : "Unknown error",
        };
      }
    },
  );
}

export type ServerLogoutOptions = {
  globalLogout?: boolean;
};

export async function serverLogout(options: ServerLogoutOptions = {}) {
  return await Sentry.withServerActionInstrumentation(
    "serverLogout",
    {},
    async () => {
      try {
        const scope = options.globalLogout ? "global" : "local";
        const result = await nativeLogout(scope);

        revalidatePath("/");

        if (!result.success) {
          console.error("Error logging out:", result.error);
          return { success: false, error: result.error };
        }

        revalidatePath("/", "layout");
        return { success: true };
      } catch (error) {
        console.error("Logout error:", error);
        return {
          success: false,
          error: error instanceof Error ? error.message : "Unknown error",
        };
      }
    },
  );
}

export async function refreshSession() {
  return await Sentry.withServerActionInstrumentation(
    "refreshSession",
    {},
    async () => {
      try {
        const result = await serverRefreshToken();

        if (!result.success || !result.user) {
          return {
            user: null,
            error: result.error,
          };
        }

        revalidatePath("/", "layout");

        return {
          user: {
            id: result.user.id,
            email: result.user.email,
            role: result.user.role,
          }
        };
      } catch (error) {
        console.error("Refresh session error:", error);
        return {
          user: null,
          error: error instanceof Error ? error.message : "Unknown error",
        };
      }
    },
  );
}
```

#### Step 5: Update Middleware

Edit `src/lib/supabase/middleware.ts`:

```typescript
import { NextResponse, type NextRequest } from "next/server";
import { isAdminPage, isProtectedPage } from "./helpers";

export async function updateSession(request: NextRequest) {
  let response = NextResponse.next({ request });

  const accessToken = request.cookies.get("access_token")?.value;

  // Parse JWT to get user info (without verification - backend will verify)
  let user = null;
  let userRole = null;

  if (accessToken) {
    try {
      const payload = JSON.parse(
        Buffer.from(accessToken.split(".")[1], "base64").toString()
      );

      // Check if token is expired
      if (payload.exp && Date.now() / 1000 < payload.exp) {
        user = { id: payload.sub, email: payload.email };
        userRole = payload.role;
      }
    } catch (e) {
      // Invalid token format
      console.error("Failed to parse access token:", e);
    }
  }

  const url = request.nextUrl.clone();
  const pathname = request.nextUrl.pathname;

  // AUTH REDIRECTS
  // 1. Check if user is not authenticated but trying to access protected content
  if (!user) {
    const attemptingProtectedPage = isProtectedPage(pathname);
    const attemptingAdminPage = isAdminPage(pathname);

    if (attemptingProtectedPage || attemptingAdminPage) {
      const currentDest = url.pathname + url.search;
      url.pathname = "/login";
      url.search = `?next=${encodeURIComponent(currentDest)}`;
      return NextResponse.redirect(url);
    }
  }

  // 2. Check if user is authenticated but lacks admin role when accessing admin pages
  if (user && userRole !== "admin" && isAdminPage(pathname)) {
    url.pathname = "/marketplace";
    return NextResponse.redirect(url);
  }

  return response;
}
```

#### Step 6: Update OAuth Callback

Edit `src/app/(platform)/auth/callback/route.ts`:

```typescript
import BackendAPI from "@/lib/autogpt-server-api";
import { NextResponse } from "next/server";
import { revalidatePath } from "next/cache";
import { shouldShowOnboarding } from "@/app/api/helpers";

// This route now just handles the redirect after OAuth
// The actual OAuth callback is handled by the backend at /api/auth/oauth/google/callback
export async function GET(request: Request) {
  const { searchParams, origin } = new URL(request.url);

  // Check if user is now authenticated (cookie should be set by backend)
  const cookies = request.headers.get("cookie") || "";
  const hasAccessToken = cookies.includes("access_token=");

  if (!hasAccessToken) {
    return NextResponse.redirect(`${origin}/auth/auth-code-error`);
  }

  let next = "/marketplace";

  try {
    const api = new BackendAPI();
    await api.createUser();

    if (await shouldShowOnboarding()) {
      next = "/onboarding";
      revalidatePath("/onboarding", "layout");
    } else {
      revalidatePath("/", "layout");
    }
  } catch (createUserError) {
    console.error("Error creating user:", createUserError);
    return NextResponse.redirect(`${origin}/error?message=user-creation-failed`);
  }

  // Get redirect destination from 'next' query parameter
  next = searchParams.get("next") || next;

  const forwardedHost = request.headers.get("x-forwarded-host");
  const isLocalEnv = process.env.NODE_ENV === "development";

  if (isLocalEnv) {
    return NextResponse.redirect(`${origin}${next}`);
  } else if (forwardedHost) {
    return NextResponse.redirect(`https://${forwardedHost}${next}`);
  } else {
    return NextResponse.redirect(`${origin}${next}`);
  }
}
```

#### Step 7: Update Google OAuth Provider Route

Edit `src/app/api/auth/provider/route.ts`:

```typescript
import { NextResponse } from "next/server";
import { serverGetGoogleAuthUrl } from "@/lib/auth/native-auth";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { provider, redirectTo } = body;

    if (provider !== "google") {
      return NextResponse.json(
        { error: "Unsupported provider" },
        { status: 400 }
      );
    }

    const result = await serverGetGoogleAuthUrl(redirectTo || "/marketplace");

    if (result.error) {
      return NextResponse.json(
        { error: result.error },
        { status: 500 }
      );
    }

    return NextResponse.json({ url: result.url });
  } catch (error) {
    console.error("OAuth provider error:", error);
    return NextResponse.json(
      { error: "Failed to initialize OAuth" },
      { status: 500 }
    );
  }
}
```

### 3.3 Option B: Big Bang Migration

Replace all Supabase references at once. Higher risk but faster.

1. Apply all the changes from Option A simultaneously
2. Remove `@supabase/ssr` and `@supabase/supabase-js` dependencies
3. Delete old Supabase configuration

---

## Phase 4: Cutover

### 4.1 Pre-Cutover Checklist

- [ ] Backend deployed with new auth endpoints
- [ ] Database migration applied
- [ ] Environment variables configured
- [ ] Postmark email templates verified
- [ ] Google OAuth redirect URIs updated
- [ ] Frontend changes tested in staging

### 4.2 Cutover Steps

1. **Deploy frontend changes**
2. **Verify login/signup works**
3. **Verify Google OAuth works**
4. **Verify password reset works**
5. **Run user migration script** (if not already done)

### 4.3 Rollback Plan

If issues occur:

1. Revert frontend to use Supabase client
2. Supabase Auth remains functional (keep it running for 30 days)
3. Users can still login via Supabase during rollback

---

## Phase 5: Cleanup (After 30 Days)

Once migration is stable:

1. **Remove Supabase dependencies from frontend**
   ```bash
   cd autogpt_platform/frontend
   pnpm remove @supabase/ssr @supabase/supabase-js
   ```

2. **Remove Supabase environment variables**
   - `NEXT_PUBLIC_SUPABASE_URL`
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY`
   - `SUPABASE_URL`
   - `SUPABASE_JWT_SECRET` (keep if using same key)

3. **Delete old Supabase files**
   - `src/lib/supabase/server/getServerSupabase.ts`
   - Any remaining Supabase-specific code

4. **Cancel Supabase subscription** (if applicable)

---

## Troubleshooting

### Users Can't Login

1. Check if user is marked as migrated: `migratedFromSupabase = true`
2. Check if password reset email was sent
3. Verify Postmark is configured correctly

### OAuth Not Working

1. Verify Google OAuth credentials in environment
2. Check redirect URI matches exactly
3. Look for errors in backend logs

### Token Issues

1. Ensure `JWT_VERIFY_KEY` matches the old `SUPABASE_JWT_SECRET`
2. Check token expiration
3. Verify audience claim is "authenticated"

### Admin Access Issues

1. Verify email is in `ADMIN_EMAIL_DOMAINS` or `ADMIN_EMAILS`
2. Check JWT role claim is "admin"
3. User may need to re-login to get new token with updated role

---

## API Reference

### Authentication Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/auth/signup` | POST | - | Register new user |
| `/api/auth/login` | POST | - | Login, returns tokens |
| `/api/auth/logout` | POST | Token | Logout |
| `/api/auth/refresh` | POST | Cookie | Refresh access token |
| `/api/auth/me` | GET | Token | Get current user |
| `/api/auth/password/reset` | POST | - | Request reset email |
| `/api/auth/password/set` | POST | - | Set new password |
| `/api/auth/verify-email` | GET | - | Verify email |
| `/api/auth/oauth/google/authorize` | GET | - | Get Google OAuth URL |
| `/api/auth/oauth/google/callback` | GET | - | OAuth callback |

### Admin Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/auth/admin/users` | GET | Admin | List users |
| `/api/auth/admin/users/{id}` | GET | Admin | Get user details |
| `/api/auth/admin/users/{id}/impersonate` | POST | Admin | Get impersonation token |
| `/api/auth/admin/users/{id}/force-password-reset` | POST | Admin | Force password reset |
| `/api/auth/admin/users/{id}/revoke-sessions` | POST | Admin | Revoke all sessions |

### Cookie Structure

| Cookie | HttpOnly | Path | Purpose |
|--------|----------|------|---------|
| `access_token` | No | `/` | JWT for API auth |
| `refresh_token` | Yes | `/api/auth/refresh` | Session refresh |

### JWT Claims

```json
{
  "sub": "user-uuid",
  "email": "user@example.com",
  "role": "authenticated",
  "aud": "authenticated",
  "iat": 1234567890,
  "exp": 1234571490
}
```
