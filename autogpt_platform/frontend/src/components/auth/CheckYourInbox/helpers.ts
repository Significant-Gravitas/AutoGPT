// Better Auth allows 3 verification emails per IP a minute, and one went out
// the moment this screen appeared.
export const RESEND_COOLDOWN_SECONDS = 60;

// Focused when the screen replaces the form, so screen readers announce it.
export const CHECK_YOUR_INBOX_HEADING_ID = "check-your-inbox-heading";

export const CHECK_YOUR_INBOX_COPY = {
  signup: {
    title: "Check your inbox",
    action: "continue",
    backPrompt: "Wrong address?",
    backLabel: "Start again",
  },
  login: {
    title: "Verify your email to log in",
    action: "log in",
    backPrompt: "Wrong account?",
    backLabel: "Back to log in",
  },
} as const;

export type CheckYourInboxReason = keyof typeof CHECK_YOUR_INBOX_COPY;
