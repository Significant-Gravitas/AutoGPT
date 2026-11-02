export interface PendingConsent {
  id: string;
  token: string;
}

export function pendingConsent(userID?: string): PendingConsent | null {
  if (!userID || typeof window === "undefined") return null;
  try {
    const value = JSON.parse(
      sessionStorage.getItem(`pro-activation:${userID}`) ?? "null",
    );
    return value &&
      typeof value.id === "string" &&
      /^[a-f0-9]{64}$/.test(value.token)
      ? value
      : null;
  } catch {
    return null;
  }
}

export function storeConsent(userID: string, consent: PendingConsent | null) {
  try {
    if (consent)
      sessionStorage.setItem(
        `pro-activation:${userID}`,
        JSON.stringify(consent),
      );
    else sessionStorage.removeItem(`pro-activation:${userID}`);
  } catch {
    /* The server remains the source of recovery if browser storage is unavailable. */
  }
}
