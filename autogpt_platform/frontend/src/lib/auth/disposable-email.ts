import { APIError } from "better-auth/api";
import {
  acceptsMail,
  type DnsLookups,
  isDisposableEmail,
} from "./email-deliverability";
import { isEmailSuppressed } from "./email-suppression";
import {
  UNDELIVERABLE_EMAIL_CODE,
  UNDELIVERABLE_EMAIL_HINT,
  UNDELIVERABLE_EMAIL_TITLE,
} from "./undeliverable-email";

export const DISPOSABLE_EMAIL_CODE = "DISPOSABLE_EMAIL";
const DISPOSABLE_EMAIL_MESSAGE =
  "This email address can't be used to sign up. Use a permanent email address.";

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

// Cloud only: refuses a domain DNS says takes no mail (see
// email-deliverability.ts). A DNS error or timeout lets the address through.
export async function assertEmailHasMailServer(
  email: string,
  options: Options = {},
) {
  if (await acceptsMail(email, options)) return;
  throw undeliverableEmailError();
}

// Sign-up and resend, cloud only: also refuses an address the mailer has
// suppressed. Either lookup failing lets the address through.
export async function assertEmailDeliverable(
  email: string,
  options: Options = {},
) {
  const [hasMailServer, suppressed] = await Promise.all([
    acceptsMail(email, options),
    isEmailSuppressed(email),
  ]);
  if (hasMailServer && !suppressed) return;
  throw undeliverableEmailError();
}

export function undeliverableEmailError() {
  return new APIError("BAD_REQUEST", {
    code: UNDELIVERABLE_EMAIL_CODE,
    message: `${UNDELIVERABLE_EMAIL_TITLE}. ${UNDELIVERABLE_EMAIL_HINT}`,
  });
}
