import { APIError, getIp } from "better-auth/api";
import type { BetterAuthOptions } from "better-auth";
import { type AuthEmailContext, claimIPEmailSlot } from "./auth-email-cooldown";

// Each of these can email an address that has no session yet.
const IP_CAPPED_PATHS = new Set(["/sign-up/email", "/send-verification-email"]);

interface Args {
  path: string | undefined;
  headers: Headers | undefined;
  context: AuthEmailContext & { options: BetterAuthOptions };
}

/**
 * A Better Auth before-hook, so it covers the sign-up page's server action
 * (auth.api, which Better Auth's own per-IP limiter never sees) as well as the
 * HTTP routes. The count lives in the verification table, which every server
 * shares. With no IP Better Auth trusts, or if the count can't be read, the
 * request goes through: the per-address cap still holds.
 */
export async function capAuthEmailsPerIP({ path, headers, context }: Args) {
  if (!path || !IP_CAPPED_PATHS.has(path) || !headers) return;
  const ip = getIp(headers, context.options);
  if (!ip) return;
  const claimed = await claimIPEmailSlot(context, ip).catch(
    (error: unknown) => {
      console.error("Failed to check the per-IP auth email cap", {
        error: error instanceof Error ? error.message : String(error),
      });
      return true;
    },
  );
  if (!claimed) {
    throw new APIError("TOO_MANY_REQUESTS", {
      message: "Too many attempts. Please try again in a few minutes.",
    });
  }
}
