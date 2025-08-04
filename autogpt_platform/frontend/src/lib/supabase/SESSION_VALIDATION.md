# Server-Side Session Validation with httpOnly Cookies

This implementation ensures that Supabase session validation is always performed on the server side using httpOnly cookies for improved security.

## Key Features

- **httpOnly Cookies**: Session cookies are inaccessible to client-side JavaScript, preventing XSS attacks
- **Server-Side Authentication**: All API requests are authenticated on the server using httpOnly cookies
- **Automatic Request Proxying**: All BackendAPI requests are automatically proxied through server actions
- **File Upload Support**: File uploads work seamlessly with httpOnly cookie authentication
- **Zero Code Changes**: Existing BackendAPI usage continues to work without modifications
- **Cross-Tab Logout**: Logout events are still synchronized across browser tabs

## How It Works

All API requests made through `BackendAPI` are automatically proxied through server actions that:

1. Retrieve the JWT token from server-side httpOnly cookies
2. Make the authenticated request to the backend API
3. Return the response to the client

This includes both regular API calls and file uploads, all handled transparently!

## Usage

### Client Components

No changes needed! The existing `useSupabase` hook and `useBackendAPI` continue to work:

```tsx
"use client";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";

function MyComponent() {
  const { user, isLoggedIn, isUserLoading, logOut } = useSupabase();
  const api = useBackendAPI();

  if (isUserLoading) return <div>Loading...</div>;
  if (!isLoggedIn) return <div>Please log in</div>;

  // Regular API calls use secure server-side authentication
  const handleGetGraphs = async () => {
    const graphs = await api.listGraphs();
    console.log(graphs);
  };

  // File uploads also work with secure authentication
  const handleFileUpload = async (file: File) => {
    try {
      const mediaUrl = await api.uploadStoreSubmissionMedia(file);
      console.log("Uploaded:", mediaUrl);
    } catch (error) {
      console.error("Upload failed:", error);
    }
  };

  return (
    <div>
      <p>Welcome, {user?.email}!</p>
      <button onClick={handleGetGraphs}>Get Graphs</button>
      <input
        type="file"
        onChange={(e) =>
          e.target.files?.[0] && handleFileUpload(e.target.files[0])
        }
      />
      <button onClick={logOut}>Log Out</button>
    </div>
  );
}
```

### Server Components

No changes needed! Server components continue to work as before:

```tsx
import { validateSession, getCurrentUser } from "@/lib/supabase/actions";
import { redirect } from "next/navigation";

async function MyServerComponent() {
  const { user, error } = await getCurrentUser();

  if (error || !user) {
    redirect("/login");
  }

  return <div>Welcome, {user.email}!</div>;
}
```

### Server Actions

No changes needed! Server actions continue to work as before:

```tsx
"use server";
import { validateSession } from "@/lib/supabase/actions";
import BackendAPI from "@/lib/autogpt-server-api";
import { redirect } from "next/navigation";

export async function myServerAction() {
  const { user, isValid } = await validateSession("/current-path");

  if (!isValid || !user) {
    redirect("/login");
    return;
  }

  // This automatically uses secure server-side authentication
  const api = new BackendAPI();
  const graphs = await api.listGraphs();

  return graphs;
}
```

### API Calls and File Uploads

All operations use the same simple code everywhere:

```tsx
// Works the same in both client and server contexts
const api = new BackendAPI();

// Regular API requests
const graphs = await api.listGraphs();
const user = await api.createUser();
const onboarding = await api.getUserOnboarding();

// File uploads
const file = new File(["content"], "example.txt", { type: "text/plain" });
const mediaUrl = await api.uploadStoreSubmissionMedia(file);
```

## Available Server Actions

- `validateSession(currentPath)` - Validates the current session and returns user data
- `getCurrentUser()` - Gets the current user without path validation
- `serverLogout()` - Logs out the user server-side
- `refreshSession()` - Refreshes the current session

## Internal Architecture

### Request Flow

All API requests (including file uploads) follow this flow:

1. **Any API call**: `api.listGraphs()` or `api.uploadStoreSubmissionMedia(file)`
2. **Proxy server action**: `proxyApiRequest()` or `proxyFileUpload()` handles the request
3. **Server authentication**: Gets JWT from httpOnly cookies
4. **Backend request**: Makes authenticated request to backend API
5. **Response**: Returns data to the calling code

### File Upload Implementation

File uploads are handled through a dedicated `proxyFileUpload` server action that:

- Receives the file data as FormData on the server
- Retrieves authentication tokens from httpOnly cookies
- Forwards the authenticated upload request to the backend
- Returns the upload result to the client

## Security Benefits

1. **XSS Protection**: httpOnly cookies can't be accessed by malicious scripts
2. **CSRF Protection**: Combined with SameSite cookie settings
3. **Server-Side Validation**: Session validation always happens on the trusted server
4. **Zero Token Exposure**: JWT tokens never exposed to client-side JavaScript
5. **Zero Attack Surface**: No client-side session manipulation possible
6. **Secure File Uploads**: File uploads maintain the same security model

## Migration Notes

- **No code changes required** - all existing BackendAPI usage continues to work
- File uploads now work seamlessly with httpOnly cookies
- Cross-tab logout functionality is preserved
- WebSocket connections may need reconnection after session changes
- All requests now have consistent security behavior
