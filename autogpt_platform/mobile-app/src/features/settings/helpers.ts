// No native imports, so tests can load this without a phone.

export const HELP_URL = "https://agpt.co/docs";
// The same pages as frontend/src/lib/legal.ts.
export const TERMS_OF_USE_URL = "https://agpt.co/legal/platform-terms-of-use";
export const PRIVACY_POLICY_URL = "https://agpt.co/legal/platform-privacy-policy";

const DELETE_FALLBACK = "We couldn't delete your account. Try again.";
// The deletion route's own words: a wrong email (400), impersonation (403)
// and a plan Stripe wouldn't cancel (502). Anything else is not for people.
const PHRASED_STATUSES = new Set([400, 403, 502]);

export function deleteErrorCopy(error: unknown) {
  const { status, message } = (error ?? {}) as { status?: unknown; message?: unknown };
  if (typeof status === "number" && PHRASED_STATUSES.has(status) && typeof message === "string" && message) {
    return message;
  }
  return DELETE_FALLBACK;
}

/** The line shown once the account is closed, as the web words it. */
export function closedCopy(eraseAfter: string | null) {
  if (!eraseAfter) {
    return "Changed your mind? Sign in on the web within 14 days and choose Keep my account.";
  }
  const day = new Date(eraseAfter).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
  return `Changed your mind? Sign in on the web before ${day} and choose Keep my account.`;
}
