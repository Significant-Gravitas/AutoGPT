import { environment } from "@/services/environment";

export const MARKETING_OPT_OUT_COOKIE = "agpt_marketing_opt_out";
export const MARKETING_OPT_OUT_COOKIE_MAX_AGE_SECONDS = 600;

// Carries a signup-page marketing opt-out across the Google OAuth round trip:
// set here before leaving for Google, read and cleared by /auth/callback (see
// marketing-opt-out-server.ts). Only a refusal is ever written; a missing
// cookie means the person did not opt out. Clearing on the way out drops a
// refusal left over from an abandoned attempt that the person has since undone.
export function setMarketingOptOutFlag(optedOut: boolean): void {
  if (typeof document === "undefined") return;

  if (!optedOut) {
    document.cookie = `${MARKETING_OPT_OUT_COOKIE}=; Path=/; Max-Age=0`;
    return;
  }

  const secure = environment.isLocal() ? "" : "; Secure";
  document.cookie = `${MARKETING_OPT_OUT_COOKIE}=1; Path=/; Max-Age=${MARKETING_OPT_OUT_COOKIE_MAX_AGE_SECONDS}; SameSite=Lax${secure}`;
}
