import { Resolver } from "node:dns/promises";
import { environment } from "@/services/environment";
import { APIError } from "better-auth/api";
import { getDomainAndParents } from "./access-blocks";
import list from "./disposable-email-domains.json";
import { isEmailSuppressed } from "./email-suppression";
import {
  UNDELIVERABLE_EMAIL_CODE,
  UNDELIVERABLE_EMAIL_HINT,
  UNDELIVERABLE_EMAIL_TITLE,
} from "./undeliverable-email";

// Vendored from github.com/disposable-email-domains/disposable-email-domains
// (CC0); refresh with scripts/refresh-disposable-email-domains.mjs.
const DISPOSABLE_DOMAINS = new Set(list.domains);

// Temporary-inbox services that rotate domains faster than the list: any
// domain whose mail lands on one of these hosts is disposable too.
export const DISPOSABLE_MX_HOSTS = new Set(["in.mail.tm"]);

const DNS_TIMEOUT_MS = 2500;
const CACHE_TTL_MS = 10 * 60 * 1000;
const CACHE_MAX_ENTRIES = 1000;

// Codes Node's resolver gives when DNS answers that a name or record type
// does not exist. Anything else (timeout, SERVFAIL, refused) is inconclusive.
const NO_DOMAIN_CODES = new Set(["ENOTFOUND"]);
const NO_RECORDS_CODES = new Set(["ENODATA"]);

export interface DnsLookups {
  resolveMx: (domain: string) => Promise<Array<{ exchange: string }>>;
  resolve4: (domain: string) => Promise<string[]>;
  resolve6: (domain: string) => Promise<string[]>;
}

interface Options {
  dns?: DnsLookups;
  timeoutMs?: number;
}

const cache = new Map<string, { value: boolean; expiresAt: number }>();

export function clearMailServerCache() {
  cache.clear();
}

export function getEmailDomain(email: string) {
  return getDomainAndParents(email)[0];
}

export function isDisposableEmailDomain(email: string) {
  return getDomainAndParents(email).some((domain) =>
    DISPOSABLE_DOMAINS.has(domain),
  );
}

// False only when DNS definitively says the domain takes no mail: it does not
// exist, publishes a null MX (RFC 7505), or has neither MX nor an A/AAAA
// record to fall back to (RFC 5321 implicit MX). Fails open on DNS errors.
export async function hasMailServer(
  domain: string,
  { dns = defaultLookups(), timeoutMs = DNS_TIMEOUT_MS }: Options = {},
) {
  const key = domain.toLowerCase().replace(/\.$/, "");
  if (!key) return false;
  const cached = cache.get(key);
  if (cached && cached.expiresAt > Date.now()) return cached.value;

  const value = await lookUpMailServer(key, dns, timeoutMs);
  if (value !== null) {
    if (cache.size >= CACHE_MAX_ENTRIES) cache.clear();
    cache.set(key, { value, expiresAt: Date.now() + CACHE_TTL_MS });
  }
  return value ?? true;
}

// One MX lookup per domain for a whole check: sign-up asks both whether the
// domain is disposable and whether it takes mail, under one deadline.
export function sharedLookups(timeoutMs = DNS_TIMEOUT_MS): DnsLookups {
  const dns = defaultLookups();
  const mx = new Map<string, Promise<Array<{ exchange: string }>>>();
  return {
    ...dns,
    resolveMx: (domain) => {
      const cached = mx.get(domain);
      if (cached) return cached;
      const answer = withDeadline(dns.resolveMx(domain), timeoutMs);
      mx.set(domain, answer);
      return answer;
    },
  };
}

export async function isDisposableEmail(email: string, options: Options = {}) {
  if (process.env.AUTH_ALLOW_DISPOSABLE_EMAILS === "true") return false;
  if (isDisposableEmailDomain(email)) return true;
  const domain = getEmailDomain(email);
  if (!domain) return false;
  const { dns = defaultLookups(), timeoutMs = DNS_TIMEOUT_MS } = options;
  const mx = await lookUp(() => dns.resolveMx(domain), timeoutMs);
  return (
    mx.kind === "records" &&
    mx.records.some((record) =>
      DISPOSABLE_MX_HOSTS.has(record.exchange.toLowerCase().replace(/\.$/, "")),
    )
  );
}

// Cloud only: local and self-hosted setups sign up with test or internal
// domains (example.com publishes a null MX).
export async function acceptsMail(email: string, options: Options = {}) {
  if (!environment.isCloud()) return true;
  const domain = getEmailDomain(email);
  return Boolean(domain) && (await hasMailServer(domain, options));
}

// Cloud only: refuses a domain DNS says takes no mail. A DNS error or timeout
// lets the address through.
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
  const [takesMail, suppressed] = await Promise.all([
    acceptsMail(email, options),
    isEmailSuppressed(email),
  ]);
  if (takesMail && !suppressed) return;
  throw undeliverableEmailError();
}

export function undeliverableEmailError() {
  return new APIError("BAD_REQUEST", {
    code: UNDELIVERABLE_EMAIL_CODE,
    message: `${UNDELIVERABLE_EMAIL_TITLE}. ${UNDELIVERABLE_EMAIL_HINT}`,
  });
}

// null means DNS gave no definitive answer.
async function lookUpMailServer(
  domain: string,
  dns: DnsLookups,
  timeoutMs: number,
): Promise<boolean | null> {
  const mx = await lookUp(() => dns.resolveMx(domain), timeoutMs);
  if (mx.kind === "records") {
    if (mx.records.length === 0) return lookUpAddress(domain, dns, timeoutMs);
    const isNullMX = mx.records.every(
      (record) => record.exchange === "" || record.exchange === ".",
    );
    return !isNullMX;
  }
  if (mx.kind === "no_domain") return false;
  if (mx.kind === "no_records") return lookUpAddress(domain, dns, timeoutMs);
  return null;
}

async function lookUpAddress(
  domain: string,
  dns: DnsLookups,
  timeoutMs: number,
): Promise<boolean | null> {
  const [v4, v6] = await Promise.all([
    lookUp(() => dns.resolve4(domain), timeoutMs),
    lookUp(() => dns.resolve6(domain), timeoutMs),
  ]);
  if (hasRecords(v4) || hasRecords(v6)) return true;
  const isMissing = (result: LookupResult<unknown>) =>
    result.kind === "no_domain" ||
    result.kind === "no_records" ||
    (result.kind === "records" && result.records.length === 0);
  return isMissing(v4) && isMissing(v6) ? false : null;
}

type LookupResult<T> =
  | { kind: "records"; records: T[] }
  | { kind: "no_domain" }
  | { kind: "no_records" }
  | { kind: "unknown" };

function hasRecords<T>(result: LookupResult<T>) {
  return result.kind === "records" && result.records.length > 0;
}

async function lookUp<T>(
  query: () => Promise<T[]>,
  timeoutMs: number,
): Promise<LookupResult<T>> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timedOut = new Promise<LookupResult<T>>((resolve) => {
    timer = setTimeout(() => resolve({ kind: "unknown" }), timeoutMs);
  });
  const answer = query().then(
    (records): LookupResult<T> => ({
      kind: "records",
      records: Array.isArray(records) ? records : [],
    }),
    (error: unknown): LookupResult<T> => {
      const code = (error as { code?: unknown } | null)?.code;
      if (typeof code !== "string") return { kind: "unknown" };
      if (NO_DOMAIN_CODES.has(code)) return { kind: "no_domain" };
      if (NO_RECORDS_CODES.has(code)) return { kind: "no_records" };
      return { kind: "unknown" };
    },
  );
  try {
    return await Promise.race([answer, timedOut]);
  } finally {
    clearTimeout(timer);
  }
}

function withDeadline<T>(query: Promise<T>, timeoutMs: number) {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(
      () =>
        reject(Object.assign(new Error("DNS timeout"), { code: "ETIMEOUT" })),
      timeoutMs,
    );
    query.then(resolve, reject).finally(() => clearTimeout(timer));
  });
}

function defaultLookups(): DnsLookups {
  const resolver = new Resolver({ timeout: DNS_TIMEOUT_MS, tries: 1 });
  return {
    resolveMx: (domain) => resolver.resolveMx(domain),
    resolve4: (domain) => resolver.resolve4(domain),
    resolve6: (domain) => resolver.resolve6(domain),
  };
}
