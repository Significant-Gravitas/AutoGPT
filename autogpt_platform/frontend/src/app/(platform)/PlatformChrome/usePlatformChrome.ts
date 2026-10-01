import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { useAuth } from "@/lib/auth/hooks/useAuth";
import { matchesRoute } from "@/lib/utils";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";

import { getRouteTitle } from "./components/InsetHeaderTitle/InsetHeaderTitle";
import { useLayoutHint } from "./components/LayoutHintProvider/LayoutHintProvider";
import { persistLayoutHint, resolveNewLayout } from "./helpers";

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
  const {
    enabled: isNewLayoutEnabled,
    ready: isFlagReady,
    answered: isFlagAnswered,
  } = useFlagStatus(Flag.AUTOGPT_NEW_LAYOUT);
  const layoutHint = useLayoutHint();
  // Also initializes the auth store — required here because the tour shell
  // replaces the Navbar, which is what normally kicks off the session check.
  const { isLoggedIn, isUserLoading } = useAuth();

  // The flag vendor is client-side data that can resolve differently on the
  // server vs the client's first render, so its answer is only applied after
  // mount. Until then the server and the first client paint both render from
  // the layout cookie the server read (see `LayoutHintProvider`) — or, for a
  // browser without one, a neutral frame with no navigation at all. The
  // classic shell is never shown on speculation: it would paint the retired
  // top nav for every sidebar user on every hard load.
  const [isMounted, setIsMounted] = useState(false);
  useEffect(() => setIsMounted(true), []);

  const isLayoutAnswered = isMounted && isFlagAnswered;
  const resolvedNewLayout = resolveNewLayout({
    enabled: Boolean(isNewLayoutEnabled),
    answered: isLayoutAnswered,
    ready: isMounted && isFlagReady,
    hint: layoutHint,
  });

  // Remember the vendor's answer (never a timeout fallback) for the next hard
  // load, so the server paints the right shell straight away.
  useEffect(() => {
    if (!isLayoutAnswered) return;
    persistLayoutHint(Boolean(isNewLayoutEnabled));
  }, [isLayoutAnswered, isNewLayoutEnabled]);

  const isExcludedRoute = NEW_LAYOUT_EXCLUDED_PREFIXES.some((prefix) =>
    matchesRoute(pathname, prefix),
  );

  const isMarketplaceRoute = matchesRoute(pathname, "/marketplace");

  const isCopilotRoute =
    matchesRoute(pathname, "/home") || matchesRoute(pathname, "/copilot");

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

  // The flag on its own, independent of the per-route exclusions, so shells
  // for excluded routes (e.g. settings) can still gate their new-layout chrome.
  const isNewLayoutActive = resolvedNewLayout === true;

  return {
    showNewLayout: isNewLayoutActive && !isExcludedRoute && !showTourSidebar,
    isNewLayoutActive,
    // Neither shell is known yet: no cookie and no vendor answer. Consumers
    // that render classic-only chrome (top Navbar, copilot's ChatSidebar)
    // must hold it back rather than treat "not new" as "classic".
    isLayoutPending: resolvedNewLayout === undefined,
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
