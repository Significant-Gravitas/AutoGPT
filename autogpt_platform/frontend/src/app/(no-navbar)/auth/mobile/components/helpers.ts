import { MOBILE_AUTH_CALLBACK } from "@/lib/auth/mobile-auth-helpers";

export function readMobileCallback(value: unknown, state: string) {
  if (!value || typeof value !== "object" || !("url" in value)) return null;
  if (typeof value.url !== "string") return null;
  try {
    const url = new URL(value.url);
    if (
      `${url.protocol}//${url.host}${url.pathname}` !== MOBILE_AUTH_CALLBACK
    ) {
      return null;
    }
    if (url.username || url.password || url.hash) return null;
    if (url.searchParams.size !== 2) return null;
    if (url.searchParams.get("state") !== state) return null;
    if (!/^[A-Za-z0-9_-]{43}$/.test(url.searchParams.get("code") ?? "")) {
      return null;
    }
    return url.toString();
  } catch {
    return null;
  }
}
