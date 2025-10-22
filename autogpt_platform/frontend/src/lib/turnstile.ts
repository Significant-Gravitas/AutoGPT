import { environment } from "@/services/environment";

export async function verifyTurnstileToken(
  token: string,
  action?: string,
): Promise<boolean> {
  if (!environment.isCAPTCHAEnabled() || environment.isLocal()) {
    return true;
  }

  try {
    const response = await fetch(
      `${environment.getAGPTServerApiUrl()}/turnstile/verify`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          token,
          action,
        }),
      },
    );

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
