import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

import { getRouteTitle } from "./components/InsetHeaderTitle/InsetHeaderTitle";

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
  // Also initializes the auth store — required here because the tour shell
  // replaces the Navbar, which is what normally kicks off the session check.
  const { isLoggedIn, isUserLoading } = useSupabase();

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

  const isMarketplaceRoute =
    pathname === "/marketplace" ||
    Boolean(pathname?.startsWith("/marketplace/"));

  const isCopilotRoute =
    pathname === "/copilot" || Boolean(pathname?.startsWith("/copilot/"));

  // Logged-out marketplace visitors get the tour demo sidebar as an upsell.
  // Waits for the session check so it never flashes at logged-in users.
  const showTourSidebar =
    isMounted && isMarketplaceRoute && !isUserLoading && !isLoggedIn;

  return {
    showNewLayout:
      isMounted && isNewLayoutEnabled && !isExcludedRoute && !showTourSidebar,
    // On copilot the inset header floats over the chat instead of stacking
    // above it, so messages scroll to the viewport top.
    overlayInsetHeader: isCopilotRoute,
    // Titleless pages collapse the header on desktop so content doesn't sit
    // below an empty strip; on mobile it stays for the sidebar trigger.
    hasInsetHeaderTitle: Boolean(getRouteTitle(pathname)),
    showTourSidebar,
  };
}
