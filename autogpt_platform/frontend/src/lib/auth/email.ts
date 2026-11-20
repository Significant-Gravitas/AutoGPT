import type { AuthEmailRequest } from "@/app/api/__generated__/models/authEmailRequest";
import { environment } from "@/services/environment";
import { AuthEmailNotSent } from "./auth-email-not-sent";
import { APIError } from "better-auth/api";
import {
  assertEmailHasMailServer,
  undeliverableEmailError,
} from "./email-deliverability";
import { mintServiceToken } from "./service-token";
import type { SignInProvider } from "./sign-in-code";

export type AuthLinkEmailType =
  | "reset_password"
  | "verify_email"
  | "verify_new_email"
  | "change_email"
  | "set_password";

type SendAuthEmailArgs =
  | {
      to: string;
      type: AuthLinkEmailType;
      url: string;
      // change_email only: the address the account is moving to.
      newEmail?: string;
    }
  // The code that confirms a provider sign-in's address (require-sign-in-code.ts).
  | {
      to: string;
      type: "sign_in_code";
      code: string;
      provider: SignInProvider;
    }
  // Tells the account's owner a provider can now sign in to it.
  | { to: string; type: "sign_in_method_added"; provider: SignInProvider };

/**
 * Delivers a Better Auth transactional email (password reset, verification,
 * email change, a sign-in code or notice) by POSTing to the backend, which
 * owns the Postmark credential and builds the message from `type`. The call is
 * authenticated with a short-lived frontend service token signed by the
 * Better Auth JWKS key the backend already trusts — no shared secret to
 * configure on either side.
 *
 * In production a failed mint/send throws so an "email sent" UI can't lie
 * about it; outside production the action link or code is logged to the
 * server console instead (local backends usually have no Postmark credential).
 * An address mail can't reach (a domain DNS says takes no mail, or one the
 * mail provider refuses) throws the undeliverable error in every environment,
 * and the resend button shows it. Disposable domains still get their email:
 * older accounts on one need it to reset a password or move to a real address.
 */
export async function sendAuthEmail(
  args: SendAuthEmailArgs,
  // Cancels the request to the backend, e.g. when a caller stops waiting.
  { signal }: { signal?: AbortSignal } = {},
) {
  const { to, type } = args;
  // DNS errors still send.
  await assertEmailHasMailServer(to);
  try {
    const token = await mintServiceToken("auth-email:send");
    const endpoint = `${environment.getAGPTServerBaseUrl()}/api/auth/email/send`;
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(requestBody(args)),
      signal,
    });

    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      if (isUndeliverable(response.status, detail)) {
        throw undeliverableEmailError();
      }
      const message =
        `Backend auth-email send failed (${response.status}) for "${type}"` +
        (detail ? `: ${detail.slice(0, 200)}` : "");
      // A 4xx refused the request; a 5xx can come after the mail provider
      // took the email, so it doesn't say the email didn't go out.
      throw response.status < 500
        ? new AuthEmailNotSent(message)
        : new Error(message);
    }
  } catch (error) {
    if (process.env.NODE_ENV === "production" || error instanceof APIError) {
      throw error;
    }
    console.info(
      `[auth-email] Send failed (${error}). ${type} email for ${to}:`,
    );
    console.info(`[auth-email] ${localFallback(args)}`);
  }
}

// Typed as the backend's schema, so a type or provider it would refuse
// doesn't compile.
function requestBody(args: SendAuthEmailArgs): AuthEmailRequest {
  switch (args.type) {
    case "sign_in_code":
      return {
        type: args.type,
        to: args.to,
        code: args.code,
        provider: args.provider,
      };
    case "sign_in_method_added":
      return { type: args.type, to: args.to, provider: args.provider };
    default:
      return {
        type: args.type,
        to: args.to,
        url: args.url,
        new_email: args.newEmail,
      };
  }
}

// What a local stack without a mailer prints instead of sending.
function localFallback(args: SendAuthEmailArgs) {
  switch (args.type) {
    case "sign_in_code":
      return `code ${args.code} (${args.provider})`;
    case "sign_in_method_added":
      return `${args.provider} sign-in added`;
    default:
      return args.url;
  }
}

function isUndeliverable(status: number, body: string) {
  if (status !== 422) return false;
  try {
    return JSON.parse(body).detail === "email_undeliverable";
  } catch {
    return false;
  }
}
