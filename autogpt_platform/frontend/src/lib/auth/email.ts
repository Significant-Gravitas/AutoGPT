import { environment } from "@/services/environment";
import { acceptsMail } from "./email-deliverability";
import { mintServiceToken } from "./service-token";

export type AuthEmailType =
  | "reset_password"
  | "verify_email"
  | "change_email"
  | "set_password";

interface SendAuthEmailArgs {
  to: string;
  type: AuthEmailType;
  url: string;
}

/**
 * Delivers a Better Auth transactional email (password reset, verification,
 * email change) by POSTing the action link to the backend, which owns the
 * Postmark credential and builds the message from `type`. The call is
 * authenticated with a short-lived frontend service token signed by the
 * Better Auth JWKS key the backend already trusts — no shared secret to
 * configure on either side.
 *
 * In production a failed mint/send throws so an "email sent" UI can't lie
 * about it; outside production the action link is logged to the server
 * console instead (local backends usually have no Postmark credential).
 * An address whose domain DNS says takes no mail is skipped silently, as if
 * sent, so it can't bounce. Disposable domains still get their email: older
 * accounts on one need it to reset a password or move to a real address.
 */
export async function sendAuthEmail({ to, type, url }: SendAuthEmailArgs) {
  // DNS errors still send.
  if (!(await acceptsMail(to))) {
    console.warn("[auth-email] Skipped an address that takes no mail", {
      type,
    });
    return;
  }
  try {
    const token = await mintServiceToken("auth-email:send");
    const endpoint = `${environment.getAGPTServerBaseUrl()}/api/auth/email/send`;
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
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
  } catch (error) {
    if (process.env.NODE_ENV === "production") {
      throw error;
    }
    console.info(
      `[auth-email] Send failed (${error}). ${type} link for ${to}:`,
    );
    console.info(`[auth-email] ${url}`);
  }
}
