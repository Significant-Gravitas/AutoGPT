import type { User } from "./types";

// The address the server answers with says whether the change applied. If not,
// an address another account holds was sent nothing, and we mustn't say which.
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
      ? `If ${newEmail} can be used, we've sent a confirmation link to your current email address. Your email changes once you click it.`
      : `If ${newEmail} can be used, we've sent a link there. Your email changes once you click it.`,
  };
}
