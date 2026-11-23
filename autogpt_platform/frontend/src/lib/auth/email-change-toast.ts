import type { User } from "./types";

// Whether an unverified user's change applies at once depends on the server's
// email verification setting, so the address the server answers with decides.
export function getEmailChangeToast(
  user: User,
  newEmail: string,
  emailNow: string | undefined,
) {
  if (emailNow?.toLowerCase() === newEmail.toLowerCase()) {
    return {
      title: "Email updated",
      description: `Your email is now ${newEmail}.`,
    };
  }
  return {
    title: "Confirm your new email",
    description: user.email_verified
      ? "We sent a confirmation link to your current email address. Your email changes once you click it."
      : `We sent a link to ${newEmail}. Your email changes once you click it.`,
  };
}
