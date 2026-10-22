import { APIError } from "better-auth/api";
import { Resolver } from "node:dns/promises";
import { getDomainAndParents } from "./access-blocks";
import {
  type DnsLookups,
  getEmailDomain,
  hasMailServer,
  isDisposableEmailDomain,
} from "./email-deliverability";

// Temporary-inbox services that rotate domains faster than the list: any
// domain whose mail lands on one of these hosts is refused too.
export const DISPOSABLE_MX_HOSTS = new Set(["in.mail.tm"]);

export const DISPOSABLE_EMAIL_CODE = "DISPOSABLE_EMAIL";
const DISPOSABLE_EMAIL_MESSAGE =
  "This email address can't be used to sign up. Use a permanent email address.";

export const UNDELIVERABLE_EMAIL_CODE = "UNDELIVERABLE_EMAIL";
const UNDELIVERABLE_EMAIL_MESSAGE =
  "Please use a real email address you can receive mail at.";

const MX_TIMEOUT_MS = 1500;

type ResolveMX = (domain: string) => Promise<Array<{ exchange: string }>>;

interface Options {
  resolveMX?: ResolveMX;
  timeoutMs?: number;
}

// Sign-up only. Set AUTH_ALLOW_DISPOSABLE_EMAILS=true to turn it off.
export async function assertNotDisposableEmail(
  email: string,
  { resolveMX = resolveMXWithTimeout, timeoutMs = MX_TIMEOUT_MS }: Options = {},
) {
  if (process.env.AUTH_ALLOW_DISPOSABLE_EMAILS === "true") return;
  const domains = getDomainAndParents(email);
  const listed = isDisposableEmailDomain(email);
  if (listed || (await hasDisposableMX(domains[0], resolveMX, timeoutMs))) {
    throw new APIError("FORBIDDEN", {
      code: DISPOSABLE_EMAIL_CODE,
      message: DISPOSABLE_EMAIL_MESSAGE,
    });
  }
}

// Fails open: a DNS error or a slow answer lets the sign-up through.
async function hasDisposableMX(
  domain: string | undefined,
  resolveMX: ResolveMX,
  timeoutMs: number,
) {
  if (!domain) return false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timedOut = new Promise<null>((resolve) => {
    timer = setTimeout(() => resolve(null), timeoutMs);
  });
  try {
    const records = await Promise.race([resolveMX(domain), timedOut]);
    return (records ?? []).some((record) =>
      DISPOSABLE_MX_HOSTS.has(record.exchange.toLowerCase().replace(/\.$/, "")),
    );
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

function resolveMXWithTimeout(domain: string) {
  return new Resolver({ timeout: MX_TIMEOUT_MS, tries: 1 }).resolveMx(domain);
}

// Sign-up only: refuses a domain DNS says takes no mail (see
// email-deliverability.ts). A DNS error or timeout lets the sign-up through.
export async function assertEmailHasMailServer(
  email: string,
  { dns }: { dns?: DnsLookups } = {},
) {
  const domain = getEmailDomain(email);
  if (domain && (await hasMailServer(domain, { dns }))) return;
  throw new APIError("BAD_REQUEST", {
    code: UNDELIVERABLE_EMAIL_CODE,
    message: UNDELIVERABLE_EMAIL_MESSAGE,
  });
}
