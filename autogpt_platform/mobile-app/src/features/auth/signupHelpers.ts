import {
  NO_APP_SESSION_CODE,
  type SignInFailure,
} from "@/lib/auth/sessionHelpers";

export const PASSWORD_MIN_LENGTH = 12;
export const RESEND_COOLDOWN_SECONDS = 60;

const OFFLINE = "Can't reach AutoGPT. Check your connection and try again.";
const NO_APP_SESSION =
  "This server can't sign the app in yet. Update it and try again.";
const PASSWORD_TOO_SHORT = `Your password needs at least ${PASSWORD_MIN_LENGTH} characters.`;
const ALREADY_EXISTS_CODES = new Set([
  "USER_ALREADY_EXISTS",
  "USER_ALREADY_EXISTS_USE_ANOTHER_EMAIL",
]);

export function canSubmitSignup(email: string, password: string) {
  return email.trim().length > 0 && password.length > 0;
}

export function passwordProblem(password: string) {
  return password.length < PASSWORD_MIN_LENGTH ? PASSWORD_TOO_SHORT : null;
}

/** The line under the sign-in form, by why the server refused (null: no answer). */
export function signInErrorCopy(failure: SignInFailure | null) {
  if (failure?.code === "EMAIL_NOT_VERIFIED") {
    return "Verify your email first. Open the link we sent you, then sign in here.";
  }
  if (!failure?.status) return OFFLINE;
  if (failure.status === 401) return "Wrong email or password.";
  if (failure.code === NO_APP_SESSION_CODE) return NO_APP_SESSION;
  return failure.message || "You can't sign in right now. Try again later.";
}

export function signupErrorCopy(failure: SignInFailure | null) {
  if (!failure?.status) return OFFLINE;
  if (failure.code && ALREADY_EXISTS_CODES.has(failure.code)) {
    return "An account with this email already exists. Sign in instead.";
  }
  if (failure.code === "PASSWORD_TOO_SHORT") return PASSWORD_TOO_SHORT;
  if (failure.code === NO_APP_SESSION_CODE) return NO_APP_SESSION;
  if (failure.status === 429) return "Too many tries. Wait a minute, then try again.";
  return failure.message || "You can't sign up right now. Try again later.";
}

/** The line on Check your inbox after "I've opened the link" or a resend. */
export function verifyErrorCopy(failure: SignInFailure | null) {
  if (failure?.code === "EMAIL_NOT_VERIFIED") {
    return "Not verified yet. Open the link in the email, then try again.";
  }
  return signInErrorCopy(failure);
}

export function resendLabel(cooldown: number) {
  return cooldown > 0 ? `Resend email in ${cooldown}s` : "Resend email";
}
