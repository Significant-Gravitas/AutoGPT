import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { useAuth } from "@/lib/auth/hooks/useAuth";
import { matchesRoute } from "@/lib/utils";

import { getRouteTitle } from "./components/InsetHeaderTitle/InsetHeaderTitle";

// Routes that render without the app sidebar. Login, signup and onboarding
// already live in the (no-navbar) group. Settings and admin bring their own
// sidebar shells (with a Back link); reset-password and the
// auth/error/unauthorized pages are all reachable while unauthenticated.
const SIDEBAR_EXCLUDED_PREFIXES = [
  "/settings",
  "/admin",
  "/reset-password",
  "/auth/auth-code-error",
  "/error",
  "/unauthorized",
];

export function usePlatformChrome() {
  const pathname = usePathname();
  // Also initializes the auth store — required here because the tour shell
  // replaces the app sidebar, which is what normally kicks off the session
  // check.
  const { isLoggedIn, isUserLoading } = useAuth();

  // The session is client-side state that can resolve differently on the
  // server vs the client's first render, so the tour swap only happens after
  // mount.
  const [isMounted, setIsMounted] = useState(false);
  useEffect(() => setIsMounted(true), []);

  const isExcludedRoute = SIDEBAR_EXCLUDED_PREFIXES.some((prefix) =>
    matchesRoute(pathname, prefix),
  );

  const isMarketplaceRoute = matchesRoute(pathname, "/marketplace");

  const isCopilotRoute =
    matchesRoute(pathname, "/home") || matchesRoute(pathname, "/copilot");

  const isBuilderRoute = matchesRoute(pathname, "/build");

  // Logged-out marketplace visitors get the tour demo sidebar as an upsell.
  // Waits for the session check so it never flashes at logged-in users.
  const showTourSidebar =
    isMounted && isMarketplaceRoute && !isUserLoading && !isLoggedIn;

  return {
    showAppSidebar: !isExcludedRoute && !showTourSidebar,
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
    // The builder wants the full canvas — the sidebar starts collapsed there
    // (defaultOpen seed for hard loads; BuilderSidebarAutoClose handles
    // client-side navigation).
    isBuilderRoute,
  };
}
