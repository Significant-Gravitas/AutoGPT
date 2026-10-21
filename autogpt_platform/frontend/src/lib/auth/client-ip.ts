import { environment } from "@/services/environment";
import type { BetterAuthOptions } from "better-auth";

type IPAddressOptions = NonNullable<BetterAuthOptions["advanced"]>["ipAddress"];

// The client IP for sessions, the sign-in IP log, IP blocks and per-IP auth
// limits. X-Forwarded-For is the client's only behind a proxy that overwrites
// it: Vercel's edge, or one the operator vouches for with
// TRUST_X_FORWARDED_FOR=true (single-container's nginx). Anywhere else no IP is
// recorded. Keep this rule matched with the client IP of #15703.
export function getIPAddressOptions(): IPAddressOptions {
  const trusted =
    environment.isVercel() || process.env.TRUST_X_FORWARDED_FOR === "true";
  return trusted
    ? { ipAddressHeaders: ["x-forwarded-for"] }
    : { disableIpTracking: true };
}
