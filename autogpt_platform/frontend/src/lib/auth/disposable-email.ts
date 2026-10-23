import { APIError } from "better-auth/api";
import {
  acceptsMail,
  type DnsLookups,
  isDisposableEmail,
} from "./email-deliverability";

export const DISPOSABLE_EMAIL_CODE = "DISPOSABLE_EMAIL";
const DISPOSABLE_EMAIL_MESSAGE =
  "This email address can't be used to sign up. Use a permanent email address.";

export const UNDELIVERABLE_EMAIL_CODE = "UNDELIVERABLE_EMAIL";
const UNDELIVERABLE_EMAIL_MESSAGE =
  "Please use a real email address you can receive mail at.";

interface Options {
  dns?: DnsLookups;
  timeoutMs?: number;
}

// Sign-up only. Set AUTH_ALLOW_DISPOSABLE_EMAILS=true to turn it off.
// Fails open: a DNS error or a slow answer lets the sign-up through.
export async function assertNotDisposableEmail(
  email: string,
  options: Options = {},
) {
  if (!(await isDisposableEmail(email, options))) return;
  throw new APIError("FORBIDDEN", {
    code: DISPOSABLE_EMAIL_CODE,
    message: DISPOSABLE_EMAIL_MESSAGE,
  });
}

// Sign-up only, cloud only: refuses a domain DNS says takes no mail (see
// email-deliverability.ts). A DNS error or timeout lets the sign-up through.
export async function assertEmailHasMailServer(
  email: string,
  options: Options = {},
) {
  if (await acceptsMail(email, options)) return;
  throw new APIError("BAD_REQUEST", {
    code: UNDELIVERABLE_EMAIL_CODE,
    message: UNDELIVERABLE_EMAIL_MESSAGE,
  });
}
