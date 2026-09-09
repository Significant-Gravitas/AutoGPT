import { createHash, randomBytes } from "node:crypto";
import { betterAuth } from "better-auth";
import { memoryAdapter } from "better-auth/adapters/memory";
import { admin } from "better-auth/plugins";
import { mobileAuth } from "../mobile-auth";

export const ORIGIN = "https://platform.agpt.co";
export const VERIFIER = randomBytes(32).toString("base64url");
export const CHALLENGE = createHash("sha256")
  .update(VERIFIER)
  .digest("base64url");
export const STATE = randomBytes(32).toString("base64url");

export function createTestAuth(
  allowSession?: () => boolean | Promise<boolean>,
  requireEmailVerification = false,
) {
  return betterAuth({
    baseURL: ORIGIN,
    secret: "mobile-auth-test-secret-at-least-32-characters", // pragma: allowlist secret
    database: memoryAdapter({
      user: [],
      session: [],
      account: [],
      verification: [],
    }),
    advanced: { disableOriginCheck: false },
    emailAndPassword: { enabled: true, requireEmailVerification },
    rateLimit: { enabled: false },
    session: { cookieCache: { enabled: true } },
    plugins: [admin(), mobileAuth()],
    databaseHooks: {
      session: {
        create: {
          async before() {
            if (allowSession && !(await allowSession())) return false;
          },
        },
      },
    },
  });
}

export function cookieHeader(response: Response) {
  return response.headers
    .getSetCookie()
    .map((cookie) => cookie.split(";")[0])
    .join("; ");
}
