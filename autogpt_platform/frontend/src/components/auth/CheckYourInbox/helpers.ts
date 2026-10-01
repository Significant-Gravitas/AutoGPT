// Better Auth allows 3 verification emails per IP a minute, and one went out
// the moment this screen appeared.
export const RESEND_COOLDOWN_SECONDS = 60;

export const CHECK_YOUR_INBOX_COPY = {
  signup: {
    title: "Check your inbox",
    action: "finish creating your account",
    backPrompt: "Wrong address?",
    backLabel: "Start again",
  },
  login: {
    title: "Verify your email to log in",
    action: "verify your email and log in",
    backPrompt: "Wrong account?",
    backLabel: "Back to log in",
  },
} as const;

export type CheckYourInboxReason = keyof typeof CHECK_YOUR_INBOX_COPY;
