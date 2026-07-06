import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

// Routes that must stay outside the new top-level sidebar layout. Login,
// signup and onboarding already live in the (no-navbar) group. These
// (platform) routes should not show the app sidebar — reset-password and the
// auth/error/unauthorized pages are all reachable while unauthenticated.
const NEW_LAYOUT_EXCLUDED_PREFIXES = [
  "/settings",
  "/reset-password",
  "/auth/auth-code-error",
  "/error",
  "/unauthorized",
];

export function usePlatformChrome() {
  const pathname = usePathname();
  const isNewLayoutEnabled = useGetFlag(Flag.AUTOGPT_NEW_LAYOUT);

  // The LaunchDarkly flag is client-side data that can resolve differently on
  // the server vs the client's first render. Switching the whole layout shell
  // on it directly causes a hydration mismatch, so we only apply the new
  // layout after mount — the server and first client paint always render the
  // classic shell, then we swap once the flag is known on the client.
  const [isMounted, setIsMounted] = useState(false);
  useEffect(() => setIsMounted(true), []);

  const isExcludedRoute = NEW_LAYOUT_EXCLUDED_PREFIXES.some((prefix) => {
    if (!pathname) return false;
    return pathname === prefix || pathname.startsWith(`${prefix}/`);
  });

  return {
    showNewLayout: isMounted && isNewLayoutEnabled && !isExcludedRoute,
  };
}
