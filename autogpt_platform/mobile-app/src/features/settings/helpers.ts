import { FRONTEND_URL } from "@/lib/env";

export function cookieScript(cookieHeader: string, frontendURL: string) {
  const { origin, protocol } = new URL(frontendURL);
  const flags = `; path=/; SameSite=Lax${protocol === "https:" ? "; Secure" : ""}`;
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
        `  document.cookie = ${JSON.stringify(`${name}=${value}${flags}`)};`,
      ];
    });
  // Android lets a navigation through when JS is slow to answer
  // onShouldStartLoadWithRequest, so this script can run on any page.
  return [
    `if (location.origin === ${JSON.stringify(origin)}) {`,
    ...assignments,
    "}",
    "true;",
  ].join("\n");
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
