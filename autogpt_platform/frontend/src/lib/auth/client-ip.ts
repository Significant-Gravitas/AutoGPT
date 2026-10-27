import { environment } from "@/services/environment";

// The client IP for the sign-in IP log and IP blocks. Where X-Forwarded-For is
// not trusted they get no IP, while Better Auth's per-IP rate limits still key on it.
export function getTrustedClientIP(ip: string | null | undefined) {
  return isForwardedForTrusted() ? ip || null : null;
}

// X-Forwarded-For is the client's only behind a proxy that overwrites it: Vercel's
// edge, or one the operator vouches for with TRUST_X_FORWARDED_FOR=true
// (single-container's nginx). Next's own server keeps whatever the caller sent.
export function isForwardedForTrusted() {
  return environment.isVercel() || process.env.TRUST_X_FORWARDED_FOR === "true";
}
