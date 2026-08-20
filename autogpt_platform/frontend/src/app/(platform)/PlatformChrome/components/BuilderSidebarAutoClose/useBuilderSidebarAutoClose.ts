import { useSidebar } from "@/components/ui/sidebar";
import { matchesRoute } from "@/lib/utils";
import { usePathname } from "next/navigation";
import { useEffect, useRef } from "react";

export function useBuilderSidebarAutoClose() {
  const pathname = usePathname();
  const { open, setOpen } = useSidebar();
  const isBuildRoute = matchesRoute(pathname, "/build");

  const openBeforeBuildRef = useRef<boolean | null>(null);
  const openRef = useRef(open);
  openRef.current = open;
  // setOpen's identity changes whenever `open` does, so the effect must only
  // act on route transitions — otherwise a manual re-open on /build would
  // re-trigger it and snap the sidebar shut again. Starts as null so the
  // first run can tell a hard load on /build apart from navigating into it.
  const wasBuildRouteRef = useRef<boolean | null>(null);

  useEffect(() => {
    const prev = wasBuildRouteRef.current;
    if (isBuildRoute === prev) return;
    wasBuildRouteRef.current = isBuildRoute;

    if (isBuildRoute) {
      // On a hard load straight into /build the provider was seeded closed
      // (defaultOpen), so "state before build" is the app default (open),
      // not the seeded value.
      openBeforeBuildRef.current = prev === null ? true : openRef.current;
      setOpen(false);
      return;
    }
    if (openBeforeBuildRef.current !== null) {
      setOpen(openBeforeBuildRef.current);
      openBeforeBuildRef.current = null;
    }
  }, [isBuildRoute, setOpen]);
}
