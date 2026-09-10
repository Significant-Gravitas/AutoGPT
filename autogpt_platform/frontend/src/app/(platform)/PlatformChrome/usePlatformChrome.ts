import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { useAuth } from "@/lib/auth/hooks/useAuth";
import { matchesRoute } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

import { getRouteTitle } from "./components/InsetHeaderTitle/InsetHeaderTitle";

// Routes that must stay outside the new top-level sidebar layout. Login,
// signup and onboarding already live in the (no-navbar) group. These
// (platform) routes should not show the app sidebar — reset-password and the
// auth/error/unauthorized pages are all reachable while unauthenticated, and
// /admin brings its own admin sidebar (see admin/layout.tsx).
const NEW_LAYOUT_EXCLUDED_PREFIXES = [
  "/settings",
  "/admin",
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
  const { isLoggedIn, isUserLoading } = useAuth();

  // The LaunchDarkly flag is client-side data that can resolve differently on
  // the server vs the client's first render. Switching the whole layout shell
  // on it directly causes a hydration mismatch, so we only apply the new
  // layout after mount — the server and first client paint always render the
  // classic shell, then we swap once the flag is known on the client.
  const [isMounted, setIsMounted] = useState(false);
  useEffect(() => setIsMounted(true), []);

  const isExcludedRoute = NEW_LAYOUT_EXCLUDED_PREFIXES.some((prefix) =>
    matchesRoute(pathname, prefix),
  );

  const isMarketplaceRoute = matchesRoute(pathname, "/marketplace");

  const isCopilotRoute = matchesRoute(pathname, "/copilot");

  const isBuilderRoute = matchesRoute(pathname, "/build");

  // Settings brings its own sidebar (with a Back link), so it renders without
  // the top Navbar even though it opts out of the new app-sidebar layout.
  const isSettingsRoute = matchesRoute(pathname, "/settings");

  // Admin mirrors settings under the new layout: its own settings-style
  // sidebar shell (see admin/layout.tsx), no top Navbar.
  const isAdminRoute = matchesRoute(pathname, "/admin");

  // Logged-out marketplace visitors get the tour demo sidebar as an upsell.
  // Waits for the session check so it never flashes at logged-in users.
  const showTourSidebar =
    isMounted && isMarketplaceRoute && !isUserLoading && !isLoggedIn;

  // The new layout is only active after mount (see hydration note above). This
  // is the flag on its own, independent of the per-route exclusions, so shells
  // for excluded routes (e.g. settings) can still gate their new-layout chrome.
  const isNewLayoutActive = isMounted && Boolean(isNewLayoutEnabled);

  return {
    showNewLayout: isNewLayoutActive && !isExcludedRoute && !showTourSidebar,
    isNewLayoutActive,
    // On copilot the inset header floats over the chat instead of stacking
    // above it, so messages scroll to the viewport top. Kept separate from
    // `isCopilotRoute` so a future overlay-header route doesn't inherit
    // copilot's header controls.
    overlayInsetHeader: isCopilotRoute,
    isCopilotRoute,
    // Titleless pages collapse the header on desktop so content doesn't sit
    // below an empty strip; on mobile it stays for the sidebar trigger.
    hasInsetHeaderTitle: Boolean(getRouteTitle(pathname)),
    showTourSidebar,
    isSettingsRoute,
    isAdminRoute,
    // The builder wants the full canvas — the sidebar starts collapsed there
    // (defaultOpen seed for hard loads; BuilderSidebarAutoClose handles
    // client-side navigation).
    isBuilderRoute,
  };
}
