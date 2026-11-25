// The headers Better Auth reads a caller's IP address from (auth.ts). The
// middleware passes them on, as they arrived, to the session lookups it makes.
// Kept apart from client-ip.ts, whose trust checks are server-only, so the
// edge middleware bundles nothing else for it. docs/platform/client-ip.md
// describes both uses, under "Sign-in" (what Better Auth reads) and "Admin
// pages" (what the middleware passes on): update it with any change here.
export const IP_ADDRESS_HEADERS = ["x-forwarded-for"];
