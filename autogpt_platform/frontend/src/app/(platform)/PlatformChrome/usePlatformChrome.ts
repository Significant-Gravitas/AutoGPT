import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { matchesRoute } from "@/lib/utils";

import { getRouteTitle } from "./components/InsetHeaderTitle/InsetHeaderTitle";
import { isCopilotPathname } from "./helpers";

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

// Nothing here may read the search params (directly or through useAuth): this
// runs at the platform layout root, so it would re-render the whole chrome on
// every URL-param write. See useIsChatSession.
export function usePlatformChrome() {
  const pathname = usePathname();
  // The store, not useAuth: useAuth reads the search params too. The app-root
  // providers (CredentialsProvider, OnboardingProvider) call useAuth on every
  // page, so the session check still starts in the tour shell.
  const isLoggedIn = useAuthStore((state) => Boolean(state.user));
  const isUserLoading = useAuthStore((state) => state.isUserLoading);

  // The session is client-side state that can resolve differently on the
  // server vs the client's first render, so the tour swap only happens after
  // mount.
  const [isMounted, setIsMounted] = useState(false);
  useEffect(() => setIsMounted(true), []);

  const isExcludedRoute = SIDEBAR_EXCLUDED_PREFIXES.some((prefix) =>
    matchesRoute(pathname, prefix),
  );

  const isMarketplaceRoute = matchesRoute(pathname, "/marketplace");

  const isCopilotRoute = isCopilotPathname(pathname);

  const isBuilderRoute = matchesRoute(pathname, "/build");

  // Logged-out marketplace visitors get public navigation and a trial CTA.
  // Waits for the session check so it never flashes at logged-in users.
  const showTourSidebar =
    isMounted && isMarketplaceRoute && !isUserLoading && !isLoggedIn;

  return {
    showAppSidebar: !isExcludedRoute && !showTourSidebar,
    showChangelog: isMounted && isLoggedIn && !isUserLoading,
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
