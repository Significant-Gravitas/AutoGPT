import {
  getPersistedSidebarOpen,
  useSidebar,
} from "@/components/molecules/Sidebar/Sidebar";
import { matchesRoute } from "@/lib/utils";
import { usePathname } from "next/navigation";
import { useEffect, useRef } from "react";

export function useBuilderSidebarAutoClose(openOutsideBuilder: boolean) {
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
      // A hard load into /build was seeded closed, so restore what any other
      // route would have opened with rather than that seed.
      openBeforeBuildRef.current =
        prev === null ? openOutsideBuilder : openRef.current;
      setOpen(false, { persist: false });
      return;
    }
    if (openBeforeBuildRef.current !== null) {
      // A choice made on /build is already remembered; showing it keeps a reload consistent.
      setOpen(getPersistedSidebarOpen() ?? openBeforeBuildRef.current, {
        persist: false,
      });
      openBeforeBuildRef.current = null;
    }
  }, [isBuildRoute, setOpen, openOutsideBuilder]);
}
