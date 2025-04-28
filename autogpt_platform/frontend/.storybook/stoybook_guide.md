# For Client-Side Components

When communicating with the server in client components, use the `useBackendAPI` hook. It automatically detects when running in a Storybook environment and switches to the mock client.

To provide custom mock data instead of the default values, add the `mockBackend` parameter in your stories:

```tsx
export const MyStory = {
  parameters: {
    mockBackend: {
      credits: 100,
      isAuthenticated: true,
      // Other custom mock data
    },
  },
};
```

# For Server-Side Components

The server-based Supabase client automatically switches between real requests and mock responses.

For server-side components, use the following pattern to select between backend client and mock client:

```tsx
const api = process.env.STORYBOOK ? new MockClient() : new BackendAPI();
```

You need to override specific API request methods in your mock client implementation. If you don't override a method, you can use the default methods provided by `BackendAPI` in both server-side and client-side environments.

To use custom mock data in server components, pass it directly to the `MockClient` constructor:

```tsx
const api = process.env.STORYBOOK
  ? new MockClient({ credits: 200, isAuthenticated: true })
  : new BackendAPI();
```

> Note: For client components, always use the `mockBackend` parameter in stories instead.

# Using MSW

If you haven't overridden an API request method in your mock client, you can use Mock Service Worker (MSW) to intercept HTTP requests from both the browser and Node.js environments, then respond with custom mock data:

```tsx
// In your story
export const WithMSW = {
  parameters: {
    msw: {
      handlers: [
        http.get("/api/user", () => {
          return HttpResponse.json({ name: "John", role: "admin" });
        }),
      ],
    },
  },
};
```

Currently, it doesn't have support for client-side Supabase client and custom data for Supabase server-side client. You could use MSW for both cases.
