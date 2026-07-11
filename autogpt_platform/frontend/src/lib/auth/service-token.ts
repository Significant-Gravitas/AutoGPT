import { signJWT } from "better-auth/plugins/jwt";

export const SERVICE_TOKEN_AUDIENCE = "autogpt-platform-backend";
export const FRONTEND_SERVICE_SUBJECT = "service:frontend";

type SignJWTContext = Parameters<typeof signJWT>[0];

/**
 * Mints a short-lived JWT proving "this request came from our frontend" for
 * backend endpoints that run before any user session exists (e.g. auth
 * emails). Signed with the same Better Auth JWKS key the backend already
 * trusts for user tokens (JWT_JWKS_URL), but with a distinct audience and
 * subject so user and service tokens can never stand in for each other.
 */
export async function mintServiceToken(scope: string) {
  // Dynamic import: auth.ts (via email.ts) imports this module, so a static
  // import back to ./auth would create an init cycle.
  const { auth } = await import("./auth");
  const context = await auth.$context;
  const nowSeconds = Math.floor(Date.now() / 1000);

  return signJWT({ context } as unknown as SignJWTContext, {
    options: {
      // Must match the jwt plugin config in auth.ts: on a fresh install with
      // no JWKS row yet (e.g. first signup email before any login), signJWT
      // creates the key, and the alg chosen here becomes the key user tokens
      // sign with too.
      jwks: { keyPairConfig: { alg: "ES256" } },
    },
    payload: {
      sub: FRONTEND_SERVICE_SUBJECT,
      aud: SERVICE_TOKEN_AUDIENCE,
      scope,
      iat: nowSeconds,
      exp: nowSeconds + 60,
    },
  });
}
