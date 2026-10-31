import { APIError } from "better-auth/api";

// An OAuth sign-up gets no verify link, so its provider must have verified the
// address. Only the password form and an admin create an unverified account.
const UNVERIFIED_SIGN_UP_PATHS = new Set([
  "/sign-up/email",
  "/admin/create-user",
]);

const EMAIL_NOT_VERIFIED_BY_PROVIDER_CODE = "EMAIL_NOT_VERIFIED_BY_PROVIDER";
const EMAIL_NOT_VERIFIED_BY_PROVIDER_MESSAGE =
  "Please verify your email address with your sign-in provider, then try again.";

// Called from the user.create.before hook with the endpoint creating the user.
export function assertProviderVerifiedEmail(
  user: { emailVerified?: boolean | null },
  ctx: { path?: string } | null | undefined,
) {
  if (user.emailVerified || UNVERIFIED_SIGN_UP_PATHS.has(ctx?.path ?? "")) {
    return;
  }

  throw new APIError("FORBIDDEN", {
    code: EMAIL_NOT_VERIFIED_BY_PROVIDER_CODE,
    message: EMAIL_NOT_VERIFIED_BY_PROVIDER_MESSAGE,
  });
}
