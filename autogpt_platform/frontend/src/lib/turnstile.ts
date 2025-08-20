/**
 * Utility functions for working with Cloudflare Turnstile
 */
import { BehaveAs, getBehaveAs } from "@/lib/utils";
import { getAgptServerApiUrl } from "@/lib/env-config";

export async function verifyTurnstileToken(
  token: string,
  action?: string,
): Promise<boolean> {
  // Skip verification unless explicitly enabled via environment variable
  if (process.env.NEXT_PUBLIC_TURNSTILE !== "enabled") {
    return true;
  }

  // Skip verification in local development
  const behaveAs = getBehaveAs();
  if (behaveAs !== BehaveAs.CLOUD) {
    return true;
  }

  try {
    const response = await fetch(`${getAgptServerApiUrl()}/turnstile/verify`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        token,
        action,
      }),
    });

    if (!response.ok) {
      console.error("Turnstile verification failed:", await response.text());
      return false;
    }

    const data = await response.json();
    return data.success === true;
  } catch (error) {
    console.error("Error verifying Turnstile token:", error);
    return false;
  }
}
