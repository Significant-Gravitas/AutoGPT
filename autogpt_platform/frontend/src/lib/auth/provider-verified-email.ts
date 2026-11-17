import { APIError } from "better-auth/api";
import { ACCOUNT_NOT_LINKED_CODE } from "./microsoft-login";

// An OAuth sign-up gets no verify link, so its provider must have verified the
// address. Only the password form and an admin create an unverified account.
const UNVERIFIED_SIGN_UP_PATHS = new Set([
  "/sign-up/email",
  "/admin/create-user",
]);

// Called from the user.create.before hook. The refusal is Better Auth's own
// when an account already uses the address, so it doesn't say whether one does.
export function assertProviderVerifiedEmail(
  user: { emailVerified?: boolean | null },
  ctx: { path?: string } | null | undefined,
) {
  if (user.emailVerified || UNVERIFIED_SIGN_UP_PATHS.has(ctx?.path ?? "")) {
    return;
  }

  // No message: Better Auth adds one to the redirect as `error_description`,
  // which its own `account_not_linked` doesn't carry.
  throw new APIError("FORBIDDEN", {
    code: ACCOUNT_NOT_LINKED_CODE,
    message: "",
  });
}
