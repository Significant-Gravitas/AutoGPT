/**
 * Browser stand-in for ./token (aliased in next.config.mjs). The real module
 * pulls in the Better Auth server instance (pg, nodemailer), which can't be
 * bundled client-side. Browser requests go through /api/proxy, which mints
 * the backend JWT server-side — so there is never a client-side token.
 */
export async function getBackendAuthToken(): Promise<string | null> {
  return null;
}
