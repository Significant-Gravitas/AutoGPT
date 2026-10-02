import { createEmailVerificationToken } from "better-auth/api";
import { sendAuthEmail } from "./email";
import { getEmailVerificationCallbackURL } from "./email-verification";

interface Args {
  user: { email: string; emailVerified?: boolean | null };
  baseURL: string;
  secret: string;
  expiresIn: number;
}

/**
 * With email verification required, Better Auth answers a sign-up for an
 * address that already has an account exactly like a new one (no session, no
 * error), so as not to reveal that the account exists, and the sign-up page
 * then says a link was sent. For an account still waiting on its link (a
 * second attempt, or one created before the flag) that has to be true, so a
 * fresh link goes out, as the resend button would send. A verified account
 * gets nothing here.
 *
 * Never throws: a failure must not turn this response into an error that a
 * brand-new address would not get.
 */
export async function emailRepeatSignUp({
  user,
  baseURL,
  secret,
  expiresIn,
}: Args) {
  if (user.emailVerified) return;
  try {
    const token = await createEmailVerificationToken(
      secret,
      user.email,
      undefined,
      expiresIn,
    );
    const callbackURL = encodeURIComponent(getEmailVerificationCallbackURL());
    await sendAuthEmail({
      to: user.email,
      type: "verify_email",
      url: `${baseURL}/api/auth/verify-email?token=${token}&callbackURL=${callbackURL}`,
    });
  } catch (error) {
    console.error("Failed to email a repeat sign-up its verification link", {
      error: error instanceof Error ? error.message : String(error),
    });
  }
}
