"use client";

import { usePathname, useRouter } from "next/navigation";
import { useEffect } from "react";

// The old /profile pages are superseded by /settings. Each legacy page maps to
// its closest new-settings equivalent. The new pages render under both the
// classic and the new layout, so this is deliberately not flag-gated — links
// in the wild (emails, bookmarks, the old sidebar) must land on the new
// surface regardless of which shell the user is on.
const LEGACY_TO_NEW_SETTINGS: Record<string, string> = {
  "/profile": "/settings/profile",
  "/profile/dashboard": "/settings/creator-dashboard",
  "/profile/credits": "/settings/billing",
  "/profile/integrations": "/settings/integrations",
  "/profile/settings": "/settings/account",
  "/profile/api-keys": "/settings/api-keys",
};

// /settings/oauth-apps is still a "Coming soon" placeholder, so sending users
// there would take a working page away. Drop this once it ships.
const KEEP_ON_LEGACY = new Set(["/profile/oauth-apps"]);

export function useNewSettingsRedirect() {
  const pathname = usePathname();
  const router = useRouter();

  const shouldRedirect =
    Boolean(pathname?.startsWith("/profile")) &&
    !KEEP_ON_LEGACY.has(pathname ?? "");

  const redirectTo = shouldRedirect
    ? (LEGACY_TO_NEW_SETTINGS[pathname ?? ""] ?? "/settings/profile")
    : null;

  useEffect(() => {
    if (!redirectTo) return;
    // Carry the query string and hash across so deep links keep working —
    // e.g. /profile/settings#notifications lands on the notifications card,
    // and Stripe's ?subscription=success still reaches /settings/billing.
    const { search, hash } = window.location;
    router.replace(`${redirectTo}${search}${hash}`);
  }, [redirectTo, router]);

  return { isRedirecting: Boolean(redirectTo) };
}
