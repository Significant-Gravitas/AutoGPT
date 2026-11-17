import type { GenericEndpointContext } from "@better-auth/core";
import { APIError } from "better-auth/api";
import { setSessionCookie } from "better-auth/cookies";
import { handleOAuthUserInfo } from "better-auth/oauth2";
import { expiredError } from "./sign-in-code-cookie";
import type { PendingRecord } from "./sign-in-code-store";
import { findProvider, markConfirmedByCode } from "./sign-in-provider";

// Better Auth's own sign-in for the provider account the callback saw, with
// the address now proven. It links that account to the address's account
// through every database hook, then starts the session. The
// account row gets no provider tokens: the callback's were never kept, and
// nothing reads a sign-in account's tokens (`/get-access-token` is off). The
// provider's next sign-in stores them.
export async function finishSignIn(
  ctx: GenericEndpointContext,
  record: PendingRecord,
) {
  markConfirmedByCode(ctx, record.providerId);
  const provider = await findProvider(ctx, record.providerId);
  if (!provider) throw expiredError(ctx);
  const result = await handleOAuthUserInfo(ctx, {
    userInfo: {
      id: record.accountId,
      email: record.email,
      emailVerified: true,
      name: record.name,
      image: record.image,
    },
    account: { providerId: record.providerId, accountId: record.accountId },
    // A code only joins an account. If the address's account went away since,
    // nothing is created in its place.
    disableSignUp: true,
    requireExactAccountBinding: true,
    source: { method: "oauth", oauth: { providerId: record.providerId } },
  });
  if (!result.data) {
    // Better Auth's sign-in errors, as its callback sends them to /login.
    throw new APIError("FORBIDDEN", {
      code: result.error.split(" ").join("_"),
      message: "",
    });
  }
  await setSessionCookie(ctx, result.data);
  return ctx.json({
    provider: record.providerId,
    newUser: result.isRegister === true,
  });
}
