import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

// Routes that must stay outside the new top-level sidebar layout. Login,
// signup and onboarding already live in the (no-navbar) group, so settings
// is the only (platform) route to exclude here.
const NEW_LAYOUT_EXCLUDED_PREFIXES = ["/settings"];

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

  const isExcludedRoute = NEW_LAYOUT_EXCLUDED_PREFIXES.some((prefix) =>
    pathname?.startsWith(prefix),
  );

  return {
    showNewLayout: isMounted && isNewLayoutEnabled && !isExcludedRoute,
  };
}
