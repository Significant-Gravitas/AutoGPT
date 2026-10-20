import { APIError } from "better-auth/api";
import { Resolver } from "node:dns/promises";
import { getDomainAndParents } from "./access-blocks";
import list from "./disposable-email-domains.json";

// Vendored from github.com/disposable-email-domains/disposable-email-domains
// (CC0); refresh with scripts/refresh-disposable-email-domains.mjs.
const DISPOSABLE_DOMAINS = new Set(list.domains);

// Temporary-inbox services that rotate domains faster than the list: any
// domain whose mail lands on one of these hosts is refused too.
export const DISPOSABLE_MX_HOSTS = new Set(["in.mail.tm"]);

export const DISPOSABLE_EMAIL_CODE = "DISPOSABLE_EMAIL";
const DISPOSABLE_EMAIL_MESSAGE =
  "This email address can't be used to sign up. Use a permanent email address.";

const MX_TIMEOUT_MS = 1500;

type ResolveMx = (domain: string) => Promise<Array<{ exchange: string }>>;

interface Options {
  resolveMx?: ResolveMx;
  timeoutMs?: number;
}

// Sign-up only. Set AUTH_ALLOW_DISPOSABLE_EMAILS=true to turn it off.
export async function assertNotDisposableEmail(
  email: string,
  { resolveMx = resolveMxWithTimeout, timeoutMs = MX_TIMEOUT_MS }: Options = {},
) {
  if (process.env.AUTH_ALLOW_DISPOSABLE_EMAILS === "true") return;
  const domains = getDomainAndParents(email);
  const listed = domains.some((domain) => DISPOSABLE_DOMAINS.has(domain));
  if (listed || (await hasDisposableMx(domains[0], resolveMx, timeoutMs))) {
    throw new APIError("FORBIDDEN", {
      code: DISPOSABLE_EMAIL_CODE,
      message: DISPOSABLE_EMAIL_MESSAGE,
    });
  }
}

// Fails open: a DNS error or a slow answer lets the sign-up through.
async function hasDisposableMx(
  domain: string | undefined,
  resolveMx: ResolveMx,
  timeoutMs: number,
) {
  if (!domain) return false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timedOut = new Promise<null>((resolve) => {
    timer = setTimeout(() => resolve(null), timeoutMs);
  });
  try {
    const records = await Promise.race([resolveMx(domain), timedOut]);
    return (records ?? []).some((record) =>
      DISPOSABLE_MX_HOSTS.has(record.exchange.toLowerCase().replace(/\.$/, "")),
    );
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

function resolveMxWithTimeout(domain: string) {
  return new Resolver({ timeout: MX_TIMEOUT_MS, tries: 1 }).resolveMx(domain);
}
