"use client";

import { usePlatformChrome } from "@/app/(platform)/PlatformChrome/usePlatformChrome";
import { usePathname, useRouter } from "next/navigation";
import { useEffect } from "react";

// The old /profile pages are superseded by /settings under the new layout.
// Each legacy page maps to its closest new-settings equivalent.
const LEGACY_TO_NEW_SETTINGS: Record<string, string> = {
  "/profile": "/settings/profile",
  "/profile/dashboard": "/settings/creator-dashboard",
  "/profile/credits": "/settings/billing",
  "/profile/integrations": "/settings/integrations",
  "/profile/settings": "/settings/account",
  "/profile/api-keys": "/settings/api-keys",
  "/profile/oauth-apps": "/settings/oauth-apps",
};

export function useNewSettingsRedirect() {
  const { isNewLayoutActive } = usePlatformChrome();
  const pathname = usePathname();
  const router = useRouter();

  const redirectTo = isNewLayoutActive
    ? (LEGACY_TO_NEW_SETTINGS[pathname ?? ""] ?? "/settings/profile")
    : null;

  useEffect(() => {
    if (redirectTo) router.replace(redirectTo);
  }, [redirectTo, router]);

  return { isRedirecting: Boolean(redirectTo) };
}
