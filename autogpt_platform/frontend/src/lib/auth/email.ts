import nodemailer from "nodemailer";

interface SendAuthEmailArgs {
  to: string;
  subject: string;
  text: string;
}

function getTransport() {
  const host = process.env.SMTP_HOST;
  if (!host) return null;

  return nodemailer.createTransport({
    host,
    port: Number(process.env.SMTP_PORT || 587),
    secure: process.env.SMTP_SECURE === "true",
    auth: process.env.SMTP_USER
      ? { user: process.env.SMTP_USER, pass: process.env.SMTP_PASS }
      : undefined,
  });
}

/**
 * Delivers an auth email (verification, password reset) via SMTP when
 * configured. Without SMTP_HOST the link is logged to the server console,
 * which matches the previous local-dev behavior where GoTrue auto-confirmed
 * signups and dropped mail into Inbucket.
 */
export async function sendAuthEmail({ to, subject, text }: SendAuthEmailArgs) {
  const transport = getTransport();

  if (!transport) {
    if (process.env.NODE_ENV === "production") {
      // Fail the auth flow rather than letting "Email sent" UI lie about an
      // undeliverable message. Never print one-time auth links into
      // production logs. This throws for every address equally, so it leaks
      // nothing about account existence.
      throw new Error(
        `SMTP is not configured — could not deliver "${subject}". ` +
          "Set SMTP_HOST (and friends) to enable auth email delivery.",
      );
    }
    console.info(`[auth-email] SMTP not configured. ${subject} for ${to}:`);
    console.info(`[auth-email] ${text}`);
    return;
  }

  await transport.sendMail({
    from: process.env.SMTP_FROM || "AutoGPT Platform <no-reply@localhost>",
    to,
    subject,
    text,
  });
}
