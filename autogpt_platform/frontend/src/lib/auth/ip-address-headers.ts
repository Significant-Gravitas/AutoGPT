// The headers Better Auth reads a caller's IP address from (auth.ts). The
// middleware passes them on, as they arrived, to the session lookups it makes.
// Kept apart from client-ip.ts, whose trust checks are server-only, so the
// edge middleware bundles nothing else for it. Documented in
// docs/platform/client-ip.md: update it with any change here.
export const IP_ADDRESS_HEADERS = ["x-forwarded-for"];
