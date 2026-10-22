import { Resolver } from "node:dns/promises";
import { getDomainAndParents } from "./access-blocks";
import list from "./disposable-email-domains.json";

// Vendored from github.com/disposable-email-domains/disposable-email-domains
// (CC0); refresh with scripts/refresh-disposable-email-domains.mjs.
const DISPOSABLE_DOMAINS = new Set(list.domains);

const DNS_TIMEOUT_MS = 2500;
const CACHE_TTL_MS = 10 * 60 * 1000;
const CACHE_MAX_ENTRIES = 1000;

// Codes Node's resolver gives when DNS answers that a name or record type
// does not exist. Anything else (timeout, SERVFAIL, refused) is inconclusive.
const NO_DOMAIN_CODES = new Set(["ENOTFOUND"]);
const NO_RECORDS_CODES = new Set(["ENODATA"]);

export type EmailDeliverability =
  | { deliverable: true }
  | { deliverable: false; reason: "disposable" | "no_mail_server" };

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

export async function checkEmailDeliverable(
  email: string,
  options: Options = {},
): Promise<EmailDeliverability> {
  const allowDisposable = process.env.AUTH_ALLOW_DISPOSABLE_EMAILS === "true";
  if (!allowDisposable && isDisposableEmailDomain(email)) {
    return { deliverable: false, reason: "disposable" };
  }
  const domain = getEmailDomain(email);
  if (!domain || !(await hasMailServer(domain, options))) {
    return { deliverable: false, reason: "no_mail_server" };
  }
  return { deliverable: true };
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

function defaultLookups(): DnsLookups {
  const resolver = new Resolver({ timeout: DNS_TIMEOUT_MS, tries: 1 });
  return {
    resolveMx: (domain) => resolver.resolveMx(domain),
    resolve4: (domain) => resolver.resolve4(domain),
    resolve6: (domain) => resolver.resolve6(domain),
  };
}
