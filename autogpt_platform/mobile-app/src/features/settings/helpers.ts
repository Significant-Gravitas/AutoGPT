import { FRONTEND_URL } from "@/lib/env";

/**
 * JavaScript that copies the stored session cookies into the page before it
 * runs. Each assignment is a JSON string literal, so no cookie value can end
 * the string and run code of its own.
 */
export function cookieScript(cookieHeader: string, isSecure: boolean) {
  const flags = `; path=/; SameSite=Lax${isSecure ? "; Secure" : ""}`;
  const assignments = cookieHeader
    .split(";")
    .map((pair) => pair.trim())
    .flatMap((pair) => {
      const split = pair.indexOf("=");
      if (split <= 0) return [];
      const name = pair.slice(0, split).trim();
      const value = pair.slice(split + 1).trim();
      if (!name) return [];
      return [
        `document.cookie = ${JSON.stringify(`${name}=${value}${flags}`)};`,
      ];
    });
  return [...assignments, "true;"].join("\n");
}

export function isAllowedSettingsURL(url: string) {
  if (!isFrontendOrigin(url)) return false;
  const { pathname } = new URL(url);
  return pathname === "/settings" || pathname.startsWith("/settings/");
}

export function isFrontendOrigin(url: string) {
  try {
    return new URL(url).origin === new URL(FRONTEND_URL).origin;
  } catch {
    return false;
  }
}

export function isWebURL(url: string) {
  return /^https?:\/\//i.test(url);
}
