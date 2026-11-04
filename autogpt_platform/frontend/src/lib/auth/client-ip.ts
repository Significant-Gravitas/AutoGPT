import { findInvalidTrustedProxies } from "@better-auth/core/utils/ip";

import { environment } from "@/services/environment";

// The client IP for the sign-in IP log and IP blocks. Where X-Forwarded-For is
// not trusted they get no IP, while Better Auth's per-IP rate limits still key on it.
export function getTrustedClientIP(ip: string | null | undefined) {
  return isForwardedForTrusted() ? ip || null : null;
}

// X-Forwarded-For is the client's only behind a proxy that overwrites it (Vercel's
// edge, or one the operator vouches for with TRUST_X_FORWARDED_FOR=true, such as
// single-container's nginx) or one whose hops TRUSTED_PROXIES lists. Next's own
// server keeps whatever the caller sent.
export function isForwardedForTrusted() {
  return (
    environment.isVercel() ||
    process.env.TRUST_X_FORWARDED_FOR === "true" ||
    trustedProxies().length > 0
  );
}

// The proxies in front of us, as IP addresses or CIDR ranges: Better Auth strips
// them from the right of X-Forwarded-For to reach the client. It only warns on an
// entry it cannot parse and then ignores it, which puts every caller back in one
// rate-limit bucket, so a bad entry throws here instead; instrumentation.ts calls
// this at server start, so the server refuses to come up rather than 500 later.
export function trustedProxies() {
  const entries = (process.env.TRUSTED_PROXIES ?? "")
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
  const invalid = findInvalidTrustedProxies(entries);
  if (invalid.length > 0) {
    throw new Error(
      `TRUSTED_PROXIES takes IP addresses or CIDR ranges; cannot parse: ${invalid.join(", ")}`,
    );
  }
  return entries;
}
