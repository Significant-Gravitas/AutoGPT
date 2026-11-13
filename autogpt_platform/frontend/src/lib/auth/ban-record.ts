import { environment } from "@/services/environment";
import { APIError } from "better-auth/api";
import { SIGN_UP_BLOCKED_CODE, SIGN_UP_BLOCKED_MESSAGE } from "./access-blocks";
import { mintServiceToken } from "./service-token";

export const BAN_RECORD_SCOPE = "auth-sign-up:check";

// Sign-up waits on this, so a slow backend must not hold it for long.
const LOOKUP_TIMEOUT_MS = 3000;

// An account erased while banned leaves a ban record on the backend (a keyed
// hash of its address, backend/data/account_erasure/ban_record.py), so the
// address can't sign up again while the ban would have lasted. Refused with
// the same sentence as an admin's block, so it doesn't say which rule refused.
export async function assertNotBannedAddress(email: string) {
  if (!(await isBannedAddress(email))) return;
  throw new APIError("FORBIDDEN", {
    code: SIGN_UP_BLOCKED_CODE,
    message: SIGN_UP_BLOCKED_MESSAGE,
  });
}

// Fails open, like the access blocks: a backend that is down or slow must
// never block a sign-up.
async function isBannedAddress(email: string) {
  try {
    const token = await mintServiceToken(BAN_RECORD_SCOPE);
    const response = await fetch(
      `${environment.getAGPTServerBaseUrl()}/api/auth/email/ban-record`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ email }),
        signal: AbortSignal.timeout(LOOKUP_TIMEOUT_MS),
      },
    );
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const { banned } = (await response.json()) as { banned?: unknown };
    return banned === true;
  } catch (error) {
    console.warn("[auth] Ban record lookup failed; letting the sign-up in", {
      error: error instanceof Error ? error.message : String(error),
    });
    return false;
  }
}
