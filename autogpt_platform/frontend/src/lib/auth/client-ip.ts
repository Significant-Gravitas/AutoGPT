import { environment } from "@/services/environment";

// The client IP for the sign-in IP log and IP blocks. X-Forwarded-For is the
// client's only behind a proxy that overwrites it: Vercel's edge, or one the
// operator vouches for with TRUST_X_FORWARDED_FOR=true (single-container's
// nginx). Anywhere else they get no IP, while Better Auth's per-IP rate limits
// still key on the header. Keep this rule matched with the client IP of #15703.
export function getTrustedClientIP(ip: string | null | undefined) {
  const trusted =
    environment.isVercel() || process.env.TRUST_X_FORWARDED_FOR === "true";
  return trusted ? ip || null : null;
}
