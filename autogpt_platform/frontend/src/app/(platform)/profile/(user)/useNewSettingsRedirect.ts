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
  "/profile/api-keys": "/settings/api-keys",
};

// Legacy pages whose /settings replacement isn't ready to receive them yet —
// redirecting would take working functionality away. Each is a one-line
// removal once the new surface catches up.
//   /profile/oauth-apps → /settings/oauth-apps is a "Coming soon" placeholder.
//   /profile/settings   → /settings/account only shows notification
//     preferences behind the ``settings-notifications`` flag (AGPT staff while
//     the design is reworked), and this is exactly where every notification
//     email — plus the List-Unsubscribe header — deep-links with
//     #notifications. Drop this once the flag is on for everyone.
const KEEP_ON_LEGACY = new Set(["/profile/oauth-apps", "/profile/settings"]);

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
    // e.g. Stripe's ?subscription=success still reaches /settings/billing.
    const { search, hash } = window.location;
    router.replace(`${redirectTo}${search}${hash}`);
  }, [redirectTo, router]);

  return { isRedirecting: Boolean(redirectTo) };
}
