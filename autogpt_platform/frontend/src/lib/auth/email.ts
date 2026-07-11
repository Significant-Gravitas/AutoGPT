import { environment } from "@/services/environment";

export type AuthEmailType = "reset_password" | "verify_email" | "change_email";

interface SendAuthEmailArgs {
  to: string;
  type: AuthEmailType;
  url: string;
}

/**
 * Delivers a Better Auth transactional email (password reset, verification,
 * email change) by POSTing the action link to the backend, which owns the
 * Postmark credential and builds the message from `type`. The frontend
 * (Vercel) holds only AUTH_EMAIL_TOKEN — a scoped shared secret — never the
 * mail-provider credential.
 *
 * Without the token the link is logged to the server console in non-production
 * (local-dev convenience, matching the old SMTP-unset behavior); in production
 * a missing token throws so an "email sent" UI can't lie about an undeliverable
 * message, and a failed backend send throws for the same reason.
 */
export async function sendAuthEmail({ to, type, url }: SendAuthEmailArgs) {
  const token = process.env.AUTH_EMAIL_TOKEN;

  if (!token) {
    if (process.env.NODE_ENV === "production") {
      throw new Error(
        `AUTH_EMAIL_TOKEN is not set — could not deliver "${type}" email. ` +
          "Set it (and configure the backend mailer) to enable auth email delivery.",
      );
    }
    console.info(
      `[auth-email] AUTH_EMAIL_TOKEN not set. ${type} link for ${to}:`,
    );
    console.info(`[auth-email] ${url}`);
    return;
  }

  const endpoint = `${environment.getAGPTServerBaseUrl()}/api/auth-email/send`;
  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Auth-Email-Token": token,
    },
    body: JSON.stringify({ type, to, url }),
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(
      `Backend auth-email send failed (${response.status}) for "${type}"` +
        (detail ? `: ${detail.slice(0, 200)}` : ""),
    );
  }
}
