import { environment } from "@/services/environment";
import { mintServiceToken } from "./service-token";

// Sign-up waits on this, so a slow mailer must not hold it for long.
const LOOKUP_TIMEOUT_MS = 3000;

/**
 * Whether the backend's mailer (Postmark) has suppressed this address after a
 * hard bounce, spam complaint or manual suppression, so it won't deliver to
 * it. Cloud only, like the mail-server check. A failed or slow lookup counts
 * as not suppressed: the mailer being down must never block a sign-up.
 */
export async function isEmailSuppressed(email: string) {
  if (!environment.isCloud()) return false;
  try {
    const token = await mintServiceToken("auth-email:send");
    const response = await fetch(
      `${environment.getAGPTServerBaseUrl()}/api/auth/email/suppression`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ to: email }),
        signal: AbortSignal.timeout(LOOKUP_TIMEOUT_MS),
      },
    );
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const { suppressed } = (await response.json()) as { suppressed?: unknown };
    return suppressed === true;
  } catch (error) {
    console.warn("[auth-email] Suppression lookup failed; sending anyway", {
      error: error instanceof Error ? error.message : String(error),
    });
    return false;
  }
}
