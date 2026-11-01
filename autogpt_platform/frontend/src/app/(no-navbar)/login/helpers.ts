import type { EmailVerificationNotice } from "@/lib/auth/email-verification";

export const ACCOUNT_BANNED_NOTICE_COPY =
  "This account has been banned. If you think this is a mistake, please contact support.";

export const EMAIL_VERIFICATION_NOTICE_COPY: Record<
  EmailVerificationNotice,
  string
> = {
  expired:
    "That verification link has expired or was already used. Log in and we'll email you a new one.",
  verified: "Your email is verified. Log in to continue.",
};
